import numpy as np
import scipy as sp
import tifffile as tff
from ome_types import from_tiff
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.measure import regionprops
from skimage.transform import resize
from tensorflow.keras.models import load_model
import click
import json

MAX_NUM_CHANNELS = 52


def histogram_normalization(image, kernel_size=None):
    """
    Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).
    If one of the inputs is a constant-value array, it will
    be normalized as an array of all zeros of the same shape.
    Args:
        image (numpy.array): numpy array of phase image data with shape
            (H, W, C). Note there is no batch index here.
        kernel_size (integer): Size of kernel for CLAHE,
            defaults to 1/8 of image size.
    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    image = image.astype("float32")
    assert len(image.shape) == 3

    for channel in range(image.shape[-1]):
        X = image[..., channel]
        sample_value = X[(0,) * X.ndim]
        if (X == sample_value).all():
            image[..., channel] = np.zeros_like(X)
            continue

        X = rescale_intensity(X, out_range=(0.0, 1.0))
        X = equalize_adapthist(X, kernel_size=kernel_size)
        image[..., channel] = X
    return image


def pad_cell(X: np.ndarray, y: np.ndarray, crop_size: int):
    delta = crop_size // 2
    X = np.pad(X, ((delta, delta), (delta, delta), (0, 0)))
    y = np.pad(y, ((delta, delta), (delta, delta)))
    return X, y


def get_crop_box(centroid, delta):
    minr = int(centroid[0]) - delta
    maxr = int(centroid[0]) + delta
    minc = int(centroid[1]) - delta
    maxc = int(centroid[1]) + delta
    return np.array([minr, minc, maxr, maxc])


def get_neighbor_masks(mask, cbox, cell_idx):
    """Returns binary masks of a cell and its neighbors. This function expects padding around
    the edges, and will throw an error if you hit a wrap around."""
    minr, minc, maxr, maxc = cbox
    assert np.issubdtype(mask.dtype, np.integer) and isinstance(cell_idx, int)

    cell_view = mask[minr:maxr, minc:maxc]
    binmask_cell = (cell_view == cell_idx).astype(np.int32)

    binmask_neighbors = (cell_view != cell_idx).astype(np.int32) * (
        cell_view != 0
    ).astype(np.int32)
    return binmask_cell, binmask_neighbors


def select_raw_combine_mask(raw, mask, channel_mask):
    new_raw = raw[channel_mask]  # (C_new, H, W)
    raw_aug_mask = np.concatenate(
        [
            np.expand_dims(new_raw, axis=-1),  # (C_new, H, W, 1)
            np.tile(
                np.expand_dims(mask, axis=0), (len(new_raw), 1, 1, 1)
            ),  # (C_new, H, W, 2)
        ],
        axis=-1,
    )  # (C_new, H, W, 3)

    return raw_aug_mask


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
    config_path = Path("../model/config.yaml")
    channel_mapping_path = Path("../model/channel_mapping.yaml")
    with open(config_path, "r") as fh:
        model_config = yaml.load(fh, yaml.Loader)
    master_channel_lst = model_config["channels"]

    with open(channel_mapping_path, "r") as fh:
        channel_mapping = yaml.load(fh, yaml.Loader)

    mapper_dict = {
        "Acinar": 1,
        "Astrocyte": 2,
        "Bcell": 3,
        "BloodVesselEndothelial": 4,
        "CD4T": 5,
        "CD8T": 6,
        "Dendritic": 7,
        "EVT": 8,
        "Endocrine": 9,
        "Endothelial": 10,
        "Enterocyte": 11,
        "Epithelial": 12,
        "Fibroblast": 13,
        "GiantCell": 14,
        "Goblet": 15,
        "ICC": 16,
        "Immune": 17,
        "LymphaticEndothelial": 18,
        "Lymphocyte": 19,
        "M1Macrophage": 20,
        "M2Macrophage": 21,
        "Macrophage": 22,
        "Mast": 23,
        "Monocyte": 24,
        "Muscle": 25,
        "Myeloid": 26,
        "Myoepithelial": 27,
        "Myofibroblast": 28,
        "NK": 29,
        "NKT": 30,
        "Nerve": 31,
        "Neuroendocrine": 32,
        "Neuron": 33,
        "Neutrophil": 34,
        "Paneth": 35,
        "Plasma": 36,
        "Secretory": 37,
        "SmoothMuscle": 38,
        "Stromal": 39,
        "Tcell": 40,
        "TransitAmplifying": 41,
        "Treg": 42,
        "Tumor": 43,
        "Unknown": 44,
    }
    mapper_dict_reversed = {v: k for k, v in mapper_dict.items()}

    # Store info on channel mappings for post-evaluation
    marker_info = {}

    # Convert pipeline output image on hubmap to model input
    orig_img = tff.imread(data_file)
    # Load channel info from metadata
    img_metadata = from_tiff(data_file)
    ch_names = [ch.name for ch in img_metadata.images[0].pixels.channels]
    marker_info["img_marker_panel"] = ch_names
    marker_info["model_marker_panel"] = master_channel_lst
    channel_lst = []
    channel_mask = []
    for idx, ch in enumerate(ch_names):
        key = channel_mapping[ch]
        if key in master_channel_lst:
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
    kernel_size = 128
    crop_size = 64
    rs = 32
    # check master list against channel_lst
    assert not set(channel_lst) - set(master_channel_lst)
    # assert len(channel_lst) == class_X.shape[-1]

    ctm = load_model(model_dir, compile=False)

    # Segmentation mask. Pipeline produces four channels. The first channel is
    # the whole-cell masks, which is what we need
    pred = tff.imread(mask_path)[0, 0, ...]
    assert pred.shape == class_X.shape[:-1]

    X = histogram_normalization(class_X, kernel_size=kernel_size)

    # this is set up for one batch at a time
    y = pred
    # B, Y, X, C
    X, y = pad_cell(X, y, crop_size)

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
        delta = crop_size // 2
        cbox = get_crop_box(prop.centroid, delta)
        self_mask, neighbor_mask = get_neighbor_masks(
            y, cbox, prop.label
        )  # (H, W), (H, W)

        # yield neighbor, cbox, prop.label, int(prop.mean_intensity)

        minr, minc, maxr, maxc = cbox
        raw_patch = X[minr:maxr, minc:maxc, :]  # (H, W, C)

        raw_patch = resize(raw_patch, (rs, rs), preserve_range=True)  # (H, W, C)
        self_mask = resize(self_mask, (rs, rs), preserve_range=True)  # (H, W)
        neighbor_mask = resize(neighbor_mask, (rs, rs), preserve_range=True)  # (H, W)

        self_mask = (self_mask > 0.5).astype(np.int32)
        neighbor_mask = (neighbor_mask > 0.5).astype(np.int32)

        raw_patch = np.transpose(raw_patch, (2, 0, 1))  # (C, H, W)
        # raw_patch = np.expand_dims(raw_patch, axis=-1) # (1, C, H, W)

        mask = np.stack([self_mask, neighbor_mask], axis=-1)  # (H, W, 2)

        mask1 = mask.astype(np.float32)
        assert (mask == mask1).all()

        app = select_raw_combine_mask(raw_patch, mask1, channel_mask)

        num_channels = app.shape[0]

        # padding
        padding_length = MAX_NUM_CHANNELS - num_channels

        paddings = np.array([[0, padding_length], [0, 0], [0, 0], [0, 0]])

        app_padded = np.pad(
            app, paddings, mode="constant", constant_values=0
        )  # (MAX_NUM_CHANNELS, H, W, 3)

        channel_list_padded = channel_lst + ["None"] * padding_length

        padding_mask = np.zeros((MAX_NUM_CHANNELS, MAX_NUM_CHANNELS), dtype=np.int32)
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
        batch_size = 2000
        if (curr_cell % batch_size == 0) or (curr_cell == total_num_cells):
            appearances_list = tf.convert_to_tensor(appearances_list)
            padding_mask_lst = tf.convert_to_tensor(padding_mask_lst)
            channel_names_lst = tf.convert_to_tensor(channel_names_lst)
            label_lst = tf.convert_to_tensor(label_lst)
            real_len_lst = tf.convert_to_tensor(real_len_lst)

            inp = {
                "appearances": appearances_list,
                "channel_padding_masks": padding_mask_lst,
                "channel_names": channel_names_lst,
                "cell_idx_label": label_lst,
                "real_len": real_len_lst,
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
