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


def extract_noise_for_AMRI_IP(vol_noisy: np.ndarray) -> np.ndarray:
    if vol_noisy.ndim != 3:
        vol_noisy = np.expand_dims(vol_noisy, axis=-1)

    # Resulting vol has subject filled with 1e3 values
    vol_filled_1e3 = fill_subject(vol_noisy, fill_value=1e3)

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

    # Debug - visualize noise crop
    # if rows_crop_till > 0 or rows_crop_from > 0 and rows_crop_from < vol_filled_1e3.shape[0]:
    #     print('Debug - visualize noise crop')
    #     from matplotlib import pyplot as plt
    #     plt.imshow(s, cmap='gray')
    #     if rows_crop_till > 0:
    #         plt.axhline(rows_crop_till)
    #         print(f"rows_crop_till: {rows_crop_till}")
    #     if rows_crop_from > 0 and rows_crop_from < vol_filled_1e3.shape[0]:
    #         plt.axhline(rows_crop_from)
    #         print(f"rows_crop_from: {rows_crop_from}")
    #     plt.show()

    # Remove zero values
    nonzero = np.nonzero(noise)
    noise = np.take(noise, nonzero).squeeze()

    # # Debug - visualize noise block
    # from matplotlib import pyplot as plt
    # print('Debug - visualize noise block')
    # true_size = np.sqrt(len(noise))
    # new_size = int(np.ceil(true_size))
    # noise_padded = np.pad(noise, (new_size ** 2 - len(noise)) // 2)
    # plt.imshow(noise_padded.reshape((new_size, new_size)))
    # plt.show()

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
