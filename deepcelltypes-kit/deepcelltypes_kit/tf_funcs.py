import tensorflow as tf
import numpy as np

from .config import DCTConfig


def _int64_feature(value):
    """Returns an int32_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[tf.cast(value, tf.int64)])
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_tfr_training_example(example, celltype_mapper_table):
    """Parser for TFR data."""
    data = {
        "appearances": tf.io.FixedLenFeature([], tf.string),
        "appearances/dim0": tf.io.FixedLenFeature([], tf.int64),
        "appearances/dim1": tf.io.FixedLenFeature([], tf.int64),
        "appearances/dim2": tf.io.FixedLenFeature([], tf.int64),
        "appearances/dim3": tf.io.FixedLenFeature([], tf.int64),
        "channel_padding_masks": tf.io.FixedLenFeature([], tf.string),
        "channel_padding_masks/dim0": tf.io.FixedLenFeature([], tf.int64),
        "channel_padding_masks/dim1": tf.io.FixedLenFeature([], tf.int64),
        "channel_names": tf.io.FixedLenFeature([], tf.string),
        "dataset_name": tf.io.FixedLenFeature([], tf.string),
        "cell_idx_label": tf.io.FixedLenFeature([], tf.int64),
        # "fname": tf.io.FixedLenFeature([], tf.string),
        # "celltypes_idx": tf.io.FixedLenFeature([], tf.int64),
        "real_len": tf.io.FixedLenFeature([], tf.int64),
        # "celltypes": tf.io.FixedLenFeature([], tf.string),
        # "celltypes/dim0": tf.io.FixedLenFeature([], tf.int64),
        "celltype_str": tf.io.FixedLenFeature([], tf.string),
        "marker_positivity": tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(example, data)
    appearances = tf.reshape(
        tf.io.parse_tensor(content["appearances"], out_type=tf.float32),
        (
            content["appearances/dim0"],
            content["appearances/dim1"],
            content["appearances/dim2"],
            content["appearances/dim3"],
        ),
    )
    channel_padding_masks = tf.reshape(
        tf.io.parse_tensor(content["channel_padding_masks"], out_type=tf.int32),
        (
            content["channel_padding_masks/dim0"],
            content["channel_padding_masks/dim1"],
        ),
    )

    # add padding for class token (the first dim)
    channel_padding_masks = tf.pad(channel_padding_masks, [[1, 0], [1, 0]], "SYMMETRIC")

    channel_names = tf.io.parse_tensor(content["channel_names"], out_type=tf.string)

    celltypes_idx = celltype_mapper_table.lookup(content["celltype_str"])
    celltypes_onehot = tf.one_hot(
        celltypes_idx - 1, depth=tf.cast(celltype_mapper_table.size(), tf.int32)
    )  # idx starts from 1, while one_hot expects 0-based idx

    marker_positivity = tf.io.parse_tensor(
        content["marker_positivity"], out_type=tf.int32
    )
    marker_positivity = tf.cast(marker_positivity, tf.float32)

    X = {
        "appearances": appearances,
        "channel_padding_masks": channel_padding_masks,
        "channel_names": channel_names,
        "dataset_name": content["dataset_name"],
        # "inpaint_channel_name": "None",
        "cell_idx_label": content["cell_idx_label"],
        # "fname": content["fname"],
        "celltypes_idx": celltypes_idx,
        "real_len": content["real_len"],
        "celltype_str": content["celltype_str"],
    }
    y = {
        "celltypes": celltypes_onehot,
        # "channel_appearance": 0.0,
        "embeddings": celltypes_idx,
        "marker_positivity": marker_positivity,
    }
    return X, y


def parse_single_example(app, index, fname, ct, tissue_folder, orig_ct, channel_list):
    dct_config = DCTConfig()

    num_channels = app.shape[0]

    # padding
    padding_length = dct_config.MAX_NUM_CHANNELS - num_channels

    paddings = np.array([[0, padding_length], [0, 0], [0, 0], [0, 0]])

    app_padded = np.pad(app, paddings, mode="constant", constant_values=0)

    channel_list_padded = channel_list + ["None"] * padding_length

    padding_mask = np.zeros(
        (dct_config.MAX_NUM_CHANNELS, dct_config.MAX_NUM_CHANNELS), dtype=np.int32
    )
    padding_mask[:num_channels, :num_channels] = 1

    positive_channels = dct_config.positivity_mapping.get(ct, [0])
    positive_channels_dataset_specific = []
    if tissue_folder in dct_config.positivity_mapping_dataset_specific:
        tissue_marker_pos_dict = dct_config.positivity_mapping_dataset_specific[
            tissue_folder
        ]
        if orig_ct in tissue_marker_pos_dict:
            positive_channels_dataset_specific = tissue_marker_pos_dict[orig_ct]
    # print(positive_channels_dataset_specific)

    marker_positivity = [
        True
        if ch in positive_channels or ch in positive_channels_dataset_specific
        else False
        for ch in channel_list
    ] + [False] * padding_length
    marker_positivity = np.array(marker_positivity, dtype=np.int32)
    # print(marker_positivity)

    assert app_padded.dtype == np.float32
    assert padding_mask.dtype == np.int32
    assert marker_positivity.dtype == np.int32

    # create tfrecord example, save
    schema = {
        "appearances": _bytes_feature(serialize_array(app_padded)),
        "appearances/dim0": _int64_feature(app_padded.shape[0]),
        "appearances/dim1": _int64_feature(app_padded.shape[1]),
        "appearances/dim2": _int64_feature(app_padded.shape[2]),
        "appearances/dim3": _int64_feature(app_padded.shape[3]),
        "channel_padding_masks": _bytes_feature(serialize_array(padding_mask)),
        "channel_padding_masks/dim0": _int64_feature(padding_mask.shape[0]),
        "channel_padding_masks/dim1": _int64_feature(padding_mask.shape[1]),
        "channel_names": _bytes_feature(
            serialize_array([item.encode("utf-8") for item in channel_list_padded])
        ),
        "dataset_name": _bytes_feature(tissue_folder.encode("utf-8")),
        "cell_idx_label": _int64_feature(index),
        "fname": _bytes_feature(fname.encode("utf-8")),
        # "celltypes_idx": _int64_feature(0),
        "real_len": _int64_feature(num_channels),
        # "celltypes": _bytes_feature(
        #     serialize_array(
        #         tf.zeros((12,), dtype=tf.float32)
        #     )  # don't include background
        # ),
        # "celltypes/dim0": _int64_feature(12),
        "celltype_str": _bytes_feature(ct.encode("utf-8")),
        "orig_celltype_str": _bytes_feature(orig_ct.encode("utf-8")),
        "marker_positivity": _bytes_feature(serialize_array(marker_positivity)),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=schema))

    return ex


if __name__ == "__main__":
    parse_single_example()
