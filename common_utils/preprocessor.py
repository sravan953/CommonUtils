import numpy as np
from skimage.transform import resize as skimage_resize


def crop_fov(vol: np.ndarray, target_fov: int) -> np.ndarray:
    """
    Assumes `vol` is isotropic.
    """
    assert vol.shape[0] == vol.shape[1]  # Isotropic

    current_fov = vol.shape[0]
    half_crop = (current_fov - target_fov) // 2
    vol_cropped = vol[half_crop:-half_crop, half_crop:-half_crop]

    return vol_cropped


def mask_subject(vol: np.ndarray, return_mask: bool = False):
    vol_masked = np.zeros_like(vol)  # Filled array of 1e3
    threshold = 0.1 * (
            np.percentile(vol, 98, axis=(0, 1)) - np.percentile(vol, 2, axis=(0, 1))
    ) + np.percentile(vol, 2, axis=(0, 1))
    mask = vol > threshold
    vol_masked[mask] = vol[mask]

    if return_mask:
        return vol_masked, mask
    else:
        return vol_masked


def normalize_volume(vol: np.ndarray):
    """
    Parameters
    ----------
    vol : np.ndarray
        Input volume of shape (x, y, slices)
    """
    _min = np.min(vol)
    _max = np.max(vol)
    _range = _max - _min
    vol_normalized = (vol - _min) / _range

    return vol_normalized


def normalize_per_slice(vol: np.ndarray):
    """
    Parameters
    ----------
    vol : np.ndarray
        Input volume of shape (x, y, slices)
    """
    # Normalize slice-wise
    _min = np.min(vol, axis=(0, 1))
    _max = np.max(vol, axis=(0, 1))
    _range = _max - _min
    if len(vol.shape) != 2:  # For 3D volumes, discard invalid slices
        valid_slices = np.where(_max != _min)[0]
        vol_normalized = (vol[..., valid_slices] - _min[valid_slices]) / _range[
            valid_slices
        ]
    else:
        vol_normalized = (vol - _min) / _range

    return vol_normalized


def resize_vol(vol: np.ndarray, size: int) -> np.ndarray:
    if vol.shape[:2] != (size, size):
        vol_resized = []
        for i in range(vol.shape[-1]):
            _slice = vol[..., i]
            # vol_resized.append(cv2.resize(_slice, (size, size)))
            vol_resized.append(skimage_resize(_slice, (size, size)))
        vol_resized = np.stack(vol_resized, axis=-1)
        return vol_resized
    return vol


def standardize_volume(vol: np.ndarray):
    """
    Parameters
    ----------
    vol : np.ndarray
        Input volume of shape (x, y, slices)
    """
    _mean = np.mean(vol)
    _std = np.std(vol)
    vol_standardized = (vol - _mean) / _std

    return vol_standardized
