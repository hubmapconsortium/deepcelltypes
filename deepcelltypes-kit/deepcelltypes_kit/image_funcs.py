import numpy as np
from tqdm import tqdm
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.transform import resize, rescale
from skimage.measure import regionprops

# TODO: wrap the processing into a function too
#   and add early stopping, just in case I want to test it out


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


def combine_raw_mask(raw, mask):
    raw_aug_mask = np.concatenate(
        [
            np.expand_dims(raw, axis=-1),  # (N, C_new, H, W, 1)
            np.tile(
                np.expand_dims(mask, axis=1), (1, raw.shape[1], 1, 1, 1)
            ),  # (N, C_new, H, W, 2)
        ],
        axis=-1,
    )  # (N, C_new, H, W, 3)

    return raw_aug_mask


def patch_generator(raw, mask, cell_index, cell_type, mpp, dct_config):
    raw = np.transpose(raw, (1, 2, 0))  # (H, W, C)
    mask = mask[0]  # (H, W), only take the whole cell mask, TODO: this is not consistent across all files, fix this 

    raw = rescale(raw, dct_config.STANDARD_MPP_RESOLUTION / mpp, preserve_range=True, channel_axis=-1)

    mask = rescale(
        mask,
        dct_config.STANDARD_MPP_RESOLUTION / mpp,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.int32)

    raw = histogram_normalization(raw, kernel_size=dct_config.HIST_NORM_KERNEL_SIZE)
    raw, mask = pad_cell(raw, mask, dct_config.CROP_SIZE)

    props = regionprops(mask, cache=False)

    for prop in tqdm(props):
        idx = prop.label
        if idx == 0: # skip background 
            continue
        if len(cell_type[cell_index == idx]) == 0: # cell exists, but no label availble
            continue
        orig_ct = cell_type[cell_index == idx][0]

        delta = dct_config.CROP_SIZE // 2
        cbox = get_crop_box(prop.centroid, delta)
        self_mask, neighbor_mask = get_neighbor_masks(
            mask, cbox, prop.label
        )  # (H, W), (H, W)

        minr, minc, maxr, maxc = cbox
        raw_patch = raw[minr:maxr, minc:maxc, :]  # (H, W, C)

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
        mask_patch = np.stack([self_mask, neighbor_mask], axis=-1)  # (H, W, 2)

        yield raw_patch, mask_patch.astype(np.float32), idx, orig_ct

