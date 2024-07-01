from typing import Union

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import laplace

from common_utils import data_utils


def get_laplacian_var(vol: np.ndarray, mask_brain: bool = True, return_arr:bool=False) -> Union[float, np.ndarray]:
    if mask_brain:
        vol = data_utils.mask_subject(vol)

    laplace_var_values = []
    for i in range(vol.shape[-1]):
        img = vol[..., i]
        laplace_var_values.append(laplace(img).var())

    if return_arr:
        return laplace_var_values
    return float(np.median(laplace_var_values))


def get_local_SNR_map(
        vol: np.ndarray, window_size: int = 3, mask_brain: bool = True
) -> np.ndarray:
    # Compute variance of noise
    noise = data_utils.extract_noise(vol)
    noise_var = noise.var()

    # Compute local SNR
    vol_squared = np.square(vol)
    kernel = 1 / (window_size ** 3) * np.ones((window_size, window_size, window_size))
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

    # # Debug - visualize
    # import sass
    #
    # sass.scroll(snr_map, cmap=['jet'])

    return snr_map
