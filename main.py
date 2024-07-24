import logging
import numpy as np
import scipy as sp
import tifffile as tff
from ome_types import from_tiff
from skimage.measure import regionprops
from skimage.transform import resize, rescale
from tensorflow.keras.models import load_model
import click
import json

import tensorflow as tf

from deepcelltypes_kit.config import DCTConfig
from deepcelltypes_kit.image_funcs import (
    histogram_normalization,
    pad_cell,
    get_crop_box,
    get_neighbor_masks,
    combine_raw_mask,
)

dct_config = DCTConfig()


# NOTE: Tensorflow eats stdout and screws up flushing - need logging to undo
# the tf mess
logger = logging.getLogger(__name__)


@click.command()
@click.option("--data-dir", default=None, help="Path to hubmap data")
@click.option(
    "--image-fname",
    default=None,
    help="Filename of the multiplexed .tif image in processed CODEX format."
)
@click.option(
    "--segmask",
    default=None,
    help="Path to .tif file containing the segmentation mask."
)
def pipeline_main(data_dir, image_fname, segmask):
    import tensorflow as tf
    from pathlib import Path
    import yaml

    # Input parsing and validation
    if data_dir is None:
        raise ValueError("Specify path to CODEX dataset with --data-dir")
    data_dir = Path(data_dir)
    # TODO: Get rid of this - run on all imgs in pipeline_output/expr instead
    if image_fname is None:
        raise ValueError("Specify the name of the file, e.g. reg001_X04_Y03.tif")
    data_file = data_dir / f"pipeline_output/expr/{image_fname}"
    # TODO: Ditto
    if segmask is None:
        raise ValueError("Provide path to segmentation mask as a .tif")
    mask_path = data_dir / f"pipeline_output/mask/{segmask}"

    # Get model channels and cell types
    model_dir = Path("../model/saved_model")
    channel_mapping_path = Path("../model/channel_mapping.yaml")

    with open(channel_mapping_path, "r") as fh:
        channel_mapping = yaml.load(fh, yaml.Loader)

    mapper_dict_reversed = {v: k for k, v in dct_config.mapper_dict.items()}

    # Store info on channel mappings for post-evaluation
    marker_info = {}

    # Convert pipeline output image on hubmap to model input
    logger.info("Loading image...")
    orig_img = tff.imread(data_file).squeeze()
    logger.info("Done.")
    # Load channel info from metadata
    img_metadata = from_tiff(data_file)
    ch_names = [ch.name for ch in img_metadata.images[0].pixels.channels]
    marker_info["img_marker_panel"] = ch_names
    marker_info["model_marker_panel"] = dct_config.master_channels
    channel_lst = []
    channel_mask = []
    for idx, ch in enumerate(ch_names):
        key = channel_mapping[ch]
        if key in dct_config.master_channels:
            channel_lst.append(key)
            channel_mask.append(True)
        else:
            channel_mask.append(False)

    logger.info(channel_lst)
    logger.info(orig_img.shape)
    multiplex_img = np.asarray(orig_img).transpose(1, 2, 0)
    logger.info(f"multiplex_img: {multiplex_img.shape}")

    # Save marker info metadata
    with open("marker_info.json", "w") as fh:
        json.dump(marker_info, fh)

    logger.info("loading model...")
    ctm = load_model(model_dir, compile=False)
    logger.info("Done.")

    class_X = multiplex_img.astype(np.float32)
    logger.info(f"class_X.shape: {class_X.shape}")
    # check master list against channel_lst
    assert not set(channel_lst) - set(dct_config.master_channels)
    # assert len(channel_lst) == class_X.shape[-1]


    # Segmentation mask. Pipeline produces four channels. The first channel is
    # the whole-cell masks, which is what we need
    logger.info("Loading mask...")
    pred = tff.imread(mask_path)[0, 0, ...]
    logger.info("Done.")
    assert pred.shape == class_X.shape[:-1]

    # Extract resolution info from image metadata
    pixel_data = img_metadata.images[0].pixels
    # Model expects square pixels
    assert pixel_data.physical_size_x == pixel_data.physical_size_y
    # Assume pixel resolution in nm | TODO: handle other cases if necessary
    assert pixel_data.physical_size_x_unit.value == "nm"
    mpp = pixel_data.physical_size_x / 1000
    logger.info(f"Image metadata: mpp = {mpp:0.3f} microns")

    logger.info("Rescaling images...")
    class_X = rescale(class_X, mpp / dct_config.STANDARD_MPP_RESOLUTION, preserve_range=True, channel_axis=-1)

    pred = rescale(
        pred,
        mpp / dct_config.STANDARD_MPP_RESOLUTION,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.int32)
    logger.info("Done")


    logger.info("Normalizing images...")
    X = histogram_normalization(class_X, kernel_size=dct_config.HIST_NORM_KERNEL_SIZE)
    logger.info("Done.")

    y = pred
    # B, Y, X, C
    X, y = pad_cell(X, y, dct_config.CROP_SIZE)

    props = regionprops(y, cache=False)
    appearances_list = []
    padding_mask_lst = []
    channel_names_lst = []
    label_lst = []
    real_len_lst = []

    total_num_cells = len(props)
    logger.info("Looping over cells...")
    for prop_idx, prop in enumerate(props):
        curr_cell = prop_idx + 1
        label = prop.label
        delta = dct_config.CROP_SIZE // 2
        cbox = get_crop_box(prop.centroid, delta)
        self_mask, neighbor_mask = get_neighbor_masks(
            y, cbox, prop.label
        )  # (H, W), (H, W)

        minr, minc, maxr, maxc = cbox
        raw_patch = X[minr:maxr, minc:maxc, :]  # (H, W, C)

        raw_patch = resize(raw_patch, (dct_config.PATCH_RESIZE_SIZE, dct_config.PATCH_RESIZE_SIZE), preserve_range=True)  # (H, W, C)
        self_mask = resize(self_mask, (dct_config.PATCH_RESIZE_SIZE, dct_config.PATCH_RESIZE_SIZE), preserve_range=True)  # (H, W)
        neighbor_mask = resize(neighbor_mask, (dct_config.PATCH_RESIZE_SIZE, dct_config.PATCH_RESIZE_SIZE), preserve_range=True)  # (H, W)

        self_mask = (self_mask > 0.5).astype(np.int32)
        neighbor_mask = (neighbor_mask > 0.5).astype(np.int32)

        raw_patch = np.transpose(raw_patch, (2, 0, 1))  # (C, H, W)
        # raw_patch = np.expand_dims(raw_patch, axis=-1) # (1, C, H, W)

        mask = np.stack([self_mask, neighbor_mask], axis=-1)  # (H, W, 2)

        mask1 = mask.astype(np.float32)
        assert (mask == mask1).all()

        raw_patch_aug = np.expand_dims(raw_patch[channel_mask, ...], axis=0)
        mask_aug = np.expand_dims(mask1, axis=0)
        app = combine_raw_mask(raw_patch_aug, mask_aug)
        app = np.squeeze(app)

        num_channels = app.shape[0]

        # padding
        padding_length = dct_config.MAX_NUM_CHANNELS - num_channels

        paddings = np.array([[0, padding_length], [0, 0], [0, 0], [0, 0]])

        app_padded = np.pad(app, paddings, mode="constant", constant_values=0)

        channel_list_padded = channel_lst + ["None"] * padding_length

        padding_mask = np.zeros(
            (dct_config.MAX_NUM_CHANNELS, dct_config.MAX_NUM_CHANNELS), dtype=np.int32
        )
        padding_mask[:num_channels, :num_channels] = 1

        padding_mask = np.pad(padding_mask, [[1, 0], [1, 0]], mode="symmetric")  # for class_token

        assert app_padded.dtype == np.float32
        assert padding_mask.dtype == np.int32

        # append each of these to list, conver to tensor
        appearances_list.append(app_padded)
        padding_mask_lst.append(padding_mask)
        channel_names_lst.append(channel_list_padded)
        label_lst.append(label)
        real_len_lst.append(num_channels)

    # TODO: Make batch_size configurable?
    batch_size = 32

    # Convert to arrays
    # NOTE: tf.convert_to_tensor is orders of magnitude slower with lists
    # see tensorflow/tensorflow#44555
    appearances_arr = np.asarray(appearances_list)
    assert appearances_arr.dtype == np.float32
    padding_masks_arr = np.asarray(padding_mask_lst)
    assert padding_masks_arr.dtype == np.int32
    channel_names_arr = np.asarray(channel_names_lst)

    logger.info("Converting to tensors...")
    with tf.device("CPU:0"):
        appearances_tsr = tf.convert_to_tensor(appearances_arr)
        padding_mask_tsr = tf.convert_to_tensor(padding_masks_arr)
        channel_names_tsr = tf.convert_to_tensor(channel_names_arr)
    logger.info("Done.")
    logger.info(f"appearances: {appearances_tsr.shape}, {appearances_tsr.dtype}")
    logger.info(f"channel_padding_masks: {padding_mask_tsr.shape}, {padding_mask_tsr.dtype}")
    logger.info(f"channel_names: {channel_names_tsr.shape}, {channel_names_tsr.dtype}")

    logger.info("Creating input...")
    inp = {
        "appearances": appearances_tsr,
        "channel_padding_masks": padding_mask_tsr,
        "channel_names": channel_names_tsr,
    }
    logger.info("Done.")

    logger.info("Predicting...")
    model_output = ctm.predict(inp, batch_size=batch_size, verbose="auto")

    # Unpack batches, extracting only predictions
    logits = model_output["celltypes"]
    pred_idx = np.argmax(sp.special.softmax(logits, axis=1), axis=1)
    # NOTE: master_cell_types[0] == background
    cell_type_predictions = [
        mapper_dict_reversed[idx + 1] for idx in pred_idx
    ]  # index starts from 1

    # Save in requested format
    centroids = np.asarray([prop.centroid for prop in props])
    with open("deepcelltypes_predictions.csv", "w") as fh:
        print("ID,DeepCellTypes")
        for i, ct in enumerate(cell_type_predictions):
            lbl_idx = i + 1
            print(f"{lbl_idx},{ct}", file=fh)


if __name__ == "__main__":
    pipeline_main()
