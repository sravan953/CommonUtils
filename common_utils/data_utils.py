from typing import Union, Tuple

import numpy as np


def crop_central_50pc(vol: np.ndarray) -> np.ndarray:
    num_slices = vol.shape[-1]
    to_crop = int(0.25 * num_slices)
    central_slice = num_slices // 2
    vol_cropped = vol[..., central_slice - to_crop: central_slice + to_crop]

    return vol_cropped


def fill_subject(vol: np.ndarray, fill_value: float = 1.0) -> np.ndarray:
    """
    Masks subject and fills with `fill_value`.

    Parameters
    ==========
    vol : np.ndarray
        Input brain volume
    fill_value : float, default=1e3
        Constant fill value.

    Returns
    =======
    vol_masked : np.ndarray
           Brain segmented volume
    """
    _, mask_indices = mask_subject(vol, return_indices=True)
    vol_filled = vol.copy()
    vol_filled[mask_indices] = fill_value

    return vol_filled


def extract_noise(vol_noisy: np.ndarray) -> np.ndarray:
    # Resulting vol has subject filled with 1e3 values
    vol_filled_1e3 = fill_subject(vol_noisy, fill_value=1e3)

    # # Debug - visualize filled subject
    # import sass
    #
    # sass.scroll(vol_filled_1e3)

    noise = []
    for i in range(vol_filled_1e3.shape[-1]):  # Iterate over slices
        s = vol_filled_1e3[..., i]  # Slice
        idx_1e3 = np.argwhere(s == 1e3)  # Indices of all pixels == 1e3
        idx_1e3 = np.hsplit(idx_1e3, 2)  # Split indices into indices of [rows, cols]
        first_row = np.min(idx_1e3[0])  # First row that contains 1e3
        last_row = np.max(idx_1e3[0]) + 1  # Last row + 1 that contains 1e3
        first_row -= 5  # Buffer
        last_row += 5  # Buffer

        if first_row > 0:
            noise_top = s[:first_row]
            noise.extend(noise_top.flatten())

        if 0 < last_row < vol_filled_1e3.shape[0]:
            noise_bottom = s[last_row:]
            noise.extend(noise_bottom.flatten())

    # Remove zero values
    nonzero = np.nonzero(noise)
    noise = np.take(noise, nonzero).squeeze()

    if len(noise) == 0:
        """
        No noise was extracted from this volume, because incompatible first_row and last_row values were encountered
        in every slice. 
        Workaround: For AMRI-IP, the minimum expected input size is 256. Therefore, arbitrarily slice first and last 32 
        rows in top and bottom slices. Choose top and bottom slices because we expect to avoid as much anatomy as 
        possible.
        """
        first_row = 32
        last_row = -32
        noise_first_slice = vol_filled_1e3[:first_row, ..., 0]  # First slice
        noise_last_slice = vol_filled_1e3[last_row:, ..., -1]  # Last slice
        noise = np.stack((noise_first_slice, noise_last_slice))
        # Remove 1e3 values
        noise = noise.flatten()
        noise = noise[noise != 1e3]

    # Debug - visualize noise block
    # import sass
    # print('Debug - visualize noise block')
    # sass.scroll(noise_patches, scroll_dim=0)

    # # Debug - noise statistics
    # print(f"Mean: {np.mean(noise):.3g}, Std: {np.std(noise):.3g}, Median: {np.median(noise):.3g}")

    return noise


def mask_subject(
        vol: np.ndarray, return_indices: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Masks the subject in a magnitude image by thresholding according to [1].

    [1] Jenkinson M. (2003). Fast, automated, N-dimensional phase-unwrapping algorithm. Magnetic resonance in medicine,
    49(1), 193–197. https://doi.org/10.1002/mrm.10354

    Parameters
    ==========
    vol : np.ndarray
        Input brain volume
    return_indices : bool
        Boolean flag indicating if the indices of the subject should be returned.

    Returns
    =======
    vol_masked : np.ndarray
        Brain segmented volume
    mask_indices : np.ndarray, optional
        Indices of the subject.
    """
    vol_masked = np.zeros_like(vol)
    threshold = 0.1 * (
            np.percentile(vol, 98, axis=(0, 1)) - np.percentile(vol, 2, axis=(0, 1))
    ) + np.percentile(vol, 2, axis=(0, 1))
    mask_indices = vol >= threshold
    vol_masked[mask_indices] = 1

    if return_indices:
        return vol_masked, mask_indices
    return vol_masked
