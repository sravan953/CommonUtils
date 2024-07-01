from typing import Union

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import laplace
from skimage.draw import disk

from common_utils import data_utils


def get_laplacian_var(
    vol: np.ndarray, mask_brain: bool = True, return_arr: bool = False
) -> Union[float, np.ndarray]:
    """
    Computes the median of the variance of the Laplacian of the input volume. The entire array can be returned by
    passing `return_arr=True`.

    Parameters
    ==========
    vol: np.ndarray
        Input volume.
    mask_brain: bool, default=True
        Whether to mask the input volume with the brain mask.
    return_arr: bool, default=False
        Whether to return the Laplacian array.
    """
    if mask_brain:
        vol = data_utils.mask_subject(vol)

    laplace_var_values = []
    for i in range(vol.shape[-1]):
        img = vol[..., i]
        laplace_var_values.append(laplace(img).var())

    if return_arr:
        return laplace_var_values
    return float(np.median(laplace_var_values))


def get_local_SNR_map_for_AMRI_IP(
    vol: np.ndarray, window: int = 3, mask_brain: bool = True
) -> np.ndarray:
    if vol.ndim != 3:
        vol = np.expand_dims(vol, axis=-1)

    # Compute variance of noise
    noise = data_utils.extract_noise_for_AMRI_IP(vol)
    noise_var = np.var(noise)

    # Compute local SNR
    vol_squared = np.square(vol)
    kernel = 1 / (window**3) * np.ones((window, window, window))
    snr_map = convolve(vol_squared, kernel, mode="constant")
    snr_map /= noise_var
    snr_map -= 2
    snr_map[snr_map < 0] = 0
    snr_map = np.sqrt(snr_map)

    # Replace values <1 to avoid divide by 0 in log10
    snr_map[snr_map < 1] = 1
    snr_map = 20 * np.log10(snr_map)  # dB

    if mask_brain:
        # Mask local SNR map and zero out values outside the brain
        _, mask_indices = data_utils.mask_subject(vol, return_indices=True)
        snr_map_masked = np.zeros_like(snr_map)
        snr_map_masked[mask_indices] = snr_map[mask_indices]
        snr_map = snr_map_masked

    # # Debug - report and visualize
    # print(
    #     f"Noise var: {noise_var:.3}, local SNR map mean/median/std:  {np.mean(snr_map):.3}, {np.median(snr_map):.3},{np.std(snr_map):.3}"
    # )
    # import sass
    #
    # sass.scroll(snr_map, cmap=['jet'])

    return snr_map


def get_local_SNR_map_lowfield_phantom(
    vol: np.ndarray, mask_radius: int = 70, window: int = 3
) -> np.ndarray:
    if vol.ndim != 3:
        vol = np.expand_dims(vol, axis=-1)

    center = vol.shape[0] // 2, vol.shape[1] // 2
    circular_mask = disk(center=center, radius=mask_radius)
    mask = np.zeros(vol.shape[:2], dtype=bool)
    mask[circular_mask] = 1

    # Debug - visualize mask
    # from matplotlib import pyplot as plt
    #
    # plt.figure()
    # plt.imshow(vol[..., 10], cmap="gray")
    # plt.imshow(mask, alpha=0.5, cmap="jet")
    # plt.axis("off")
    # plt.show()

    # Compute variance of noise
    noise_crop = vol[~mask]
    noise_var = np.var(noise_crop)

    # Compute local SNR
    vol_squared = np.square(vol)
    kernel = 1 / (window**3) * np.ones((window, window, window))
    snr_map = convolve(vol_squared, kernel, mode="constant")
    snr_map /= noise_var
    snr_map -= 2
    snr_map[snr_map < 0] = 0
    snr_map = np.sqrt(snr_map)

    # Replace values <1 to avoid divide by 0 in log10
    snr_map[snr_map < 1] = 1
    snr_map = 20 * np.log10(snr_map)  # dB

    return snr_map
