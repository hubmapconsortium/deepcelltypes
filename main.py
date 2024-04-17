import numpy as np
import scipy as sp
import tifffile as tff
from ome_types import from_tiff
from skimage.measure import regionprops
from skimage.transform import resize
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

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[1:],'GPU') # only using gpu1


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
    orig_img = tff.imread(data_file).squeeze()
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

    print(channel_lst)
    print(orig_img.shape)
    multiplex_img = np.asarray(orig_img).transpose(1, 2, 0)

    # Save marker info metadata
    with open("marker_info.json", "w") as fh:
        json.dump(marker_info, fh)

    class_X = multiplex_img.astype(np.float32)
    print("class_X.shape", class_X.shape)
    # check master list against channel_lst
    assert not set(channel_lst) - set(dct_config.master_channels)
    # assert len(channel_lst) == class_X.shape[-1]

    ctm = load_model(model_dir, compile=False)

    # Segmentation mask. Pipeline produces four channels. The first channel is
    # the whole-cell masks, which is what we need
    pred = tff.imread(mask_path)[0, 0, ...]
    assert pred.shape == class_X.shape[:-1]

    X = histogram_normalization(class_X, kernel_size=dct_config.HIST_NORM_KERNEL_SIZE)

    # this is set up for one batch at a time
    y = pred
    # B, Y, X, C
    X, y = pad_cell(X, y, dct_config.CROP_SIZE)

    props = regionprops(y, cache=False)
    appearances_list = []
    padding_mask_lst = []
    channel_names_lst = []
    label_lst = []
    real_len_lst = []

    model_output = []

    total_num_cells = len(props)
    for prop_idx, prop in enumerate(props):
        curr_cell = prop_idx + 1
        label = prop.label
        delta = dct_config.CROP_SIZE // 2
        cbox = get_crop_box(prop.centroid, delta)
        self_mask, neighbor_mask = get_neighbor_masks(
            y, cbox, prop.label
        )  # (H, W), (H, W)

        # yield neighbor, cbox, prop.label, int(prop.mean_intensity)

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

        padding_mask = np.pad(padding_mask, [[1, 0], [1, 0]])  # for class_token

        assert app_padded.dtype == np.float32
        assert padding_mask.dtype == np.int32

        # append each of these to list, conver to tensor
        appearances_list.append(app_padded)
        padding_mask_lst.append(padding_mask)
        channel_names_lst.append(channel_list_padded)
        label_lst.append(label)
        real_len_lst.append(num_channels)

        # TODO: Make batch_size configurable?
        batch_size = 500
        if (curr_cell % batch_size == 0) or (curr_cell == total_num_cells):
            appearances_list = tf.convert_to_tensor(appearances_list)
            padding_mask_lst = tf.convert_to_tensor(padding_mask_lst)
            channel_names_lst = tf.convert_to_tensor(channel_names_lst)
            label_lst = tf.convert_to_tensor(label_lst)
            real_len_lst = tf.convert_to_tensor(real_len_lst)
            domain_name = tf.convert_to_tensor(["CODEX"] * len(label_lst))

            inp = {
                "appearances": appearances_list,
                "channel_padding_masks": padding_mask_lst,
                "channel_names": channel_names_lst,
                "cell_idx_label": label_lst,
                "real_len": real_len_lst,
                "domain_name": domain_name,
            }

            model_output.append(ctm.predict(inp))

            appearances_list = []
            padding_mask_lst = []
            channel_names_lst = []
            label_lst = []
            real_len_lst = []

    # Unpack batches, extracting only predictions
    logits = np.concatenate([batch["celltypes"] for batch in model_output])
    pred_idx = np.argmax(sp.special.softmax(logits, axis=1), axis=1)
    # NOTE: master_cell_types[0] == background
    cell_type_predictions = [
        mapper_dict_reversed[idx + 1] for idx in pred_idx
    ]  # index starts from 1

    # Save in requested format
    centroids = np.asarray([prop.centroid for prop in props])
    with open("deepcelltypes_predictions.csv", "w") as fh:
        for i, (centroid, ct) in enumerate(zip(centroids, cell_type_predictions)):
            lbl_idx = i + 1
            x, y = centroid
            fh.write(f"{lbl_idx},{x:.2f},{y:.2f},{ct}\n")


if __name__ == "__main__":
    pipeline_main()
