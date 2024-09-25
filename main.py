import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy as sp
import tensorflow as tf
import tifffile as tff
from deepcelltypes_kit.config import DCTConfig
from deepcelltypes_kit.image_funcs import (
    combine_raw_mask,
    get_crop_box,
    get_neighbor_masks,
    histogram_normalization,
    pad_cell,
)
from ome_types import from_tiff
from ome_utils import find_ome_tiffs, get_converted_physical_size
from skimage.measure import regionprops
from skimage.transform import rescale, resize
from tensorflow.keras.models import load_model

dct_config = DCTConfig()


# NOTE: Tensorflow eats stdout and screws up flushing - need logging to undo
# the tf mess
logger = logging.getLogger(__name__)


def predict(expr_file: Path, mask_file: Path) -> List[Tuple[int, int]]:
    import tensorflow as tf
    import yaml

    # Get model channels and cell types
    model_dir = Path("../model/saved_model")
    channel_mapping_path = Path("../model/channel_mapping.yaml")

    with open(channel_mapping_path, "r") as fh:
        channel_mapping = yaml.load(fh, yaml.Loader)

    mapper_dict_reversed = {v: k for k, v in dct_config.mapper_dict.items()}

    # Store info on channel mappings for post-evaluation
    marker_info = {}

    # Convert pipeline output image on hubmap to model input
    logger.info("Loading image %s", expr_file)
    orig_img = tff.imread(expr_file).squeeze()
    logger.info("Done.")
    # Load channel info from metadata
    img_metadata = from_tiff(expr_file)
    ch_names = [ch.name for ch in img_metadata.images[0].pixels.channels]
    marker_info["img_marker_panel"] = ch_names
    marker_info["model_marker_panel"] = dct_config.master_channels
    channel_lst = []
    channel_mask = []
    for idx, ch in enumerate(ch_names):
        key = channel_mapping.get(ch, ch)
        if key in dct_config.master_channels:
            channel_lst.append(key)
            channel_mask.append(True)
        else:
            channel_mask.append(False)

    logger.info("channel_list: %s", channel_lst)
    logger.info("orig_img.shape: %s", orig_img.shape)
    logger.info("channel_mask: %s", channel_mask)
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
    pred = tff.imread(mask_file)[0, 0, ...]
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
    class_X = rescale(
        class_X,
        mpp / dct_config.STANDARD_MPP_RESOLUTION,
        preserve_range=True,
        channel_axis=-1,
    )

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

        raw_patch = resize(
            raw_patch,
            (dct_config.PATCH_RESIZE_SIZE, dct_config.PATCH_RESIZE_SIZE),
            preserve_range=True,
        )  # (H, W, C)
        self_mask = resize(
            self_mask,
            (dct_config.PATCH_RESIZE_SIZE, dct_config.PATCH_RESIZE_SIZE),
            preserve_range=True,
        )  # (H, W)
        neighbor_mask = resize(
            neighbor_mask,
            (dct_config.PATCH_RESIZE_SIZE, dct_config.PATCH_RESIZE_SIZE),
            preserve_range=True,
        )  # (H, W)

        self_mask = (self_mask > 0.5).astype(np.int32)
        neighbor_mask = (neighbor_mask > 0.5).astype(np.int32)

        raw_patch = np.transpose(raw_patch, (2, 0, 1))  # (C, H, W)
        # raw_patch = np.expand_dims(raw_patch, axis=-1) # (1, C, H, W)

        mask = np.stack([self_mask, neighbor_mask], axis=-1)  # (H, W, 2)

        mask1 = mask.astype(np.float32)
        assert (mask == mask1).all()

        logger.info("raw_patch = %s", raw_patch)
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

        padding_mask = np.pad(
            padding_mask, [[1, 0], [1, 0]], mode="symmetric"
        )  # for class_token

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
    logger.info(
        f"channel_padding_masks: {padding_mask_tsr.shape}, {padding_mask_tsr.dtype}"
    )
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

    prediction_list = list(enumerate(cell_type_predictions, 1))
    return prediction_list


def main(data_dir: Path):
    pipeline_output_dir = data_dir / "pipeline_output"
    expr_files = sorted(find_ome_tiffs(pipeline_output_dir / "expr"))
    mask_files = sorted(find_ome_tiffs(pipeline_output_dir / "mask"))

    output_path = Path("deepcelltypes")
    output_path.mkdir(exist_ok=True, parents=True)
    for expr_file, mask_file in zip(expr_files, mask_files):
        pred_csv_file = output_path / f"{expr_file.stem}-predictions.csv"
        predictions = predict(expr_file, mask_file)
        logger.info("Saving predictions from %s to %s", expr_file, pred_csv_file)
        with open(pred_csv_file, "w") as fh:
            for idx, ct in predictions:
                print(f"{idx},{ct}", file=fh)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("data_dir", type=Path)
    args = p.parse_args()

    main(args.data_dir)
