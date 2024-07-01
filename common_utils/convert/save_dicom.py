from pathlib import Path

import numpy as np
import pydicom as pyd


def _get_dcm_vol_max(dicoms: list) -> float:
    m = 0
    for dcm in dicoms:
        m = max(m, np.max(dcm.pixel_array))

    return m


def save_vol_as_DICOMs(original_dicoms, vol: np.ndarray, path_save: Path):
    # Restore dynamic range
    # We do NOT re-normalize each slice since the entre denoised volume is normalized
    dcm_all_min = [dcm.pixel_array.min() for dcm in original_dicoms]
    dcm_all_max = [dcm.pixel_array.max() for dcm in original_dicoms]
    dcm_min = np.min(dcm_all_min)
    dcm_max = np.max(dcm_all_max)
    vol_norm = (vol * (dcm_max - dcm_min)) + dcm_min
    vol_norm = vol_norm.astype("uint16")

    for i in range(vol.shape[-1]):
        dcm = original_dicoms[i]
        s = vol_norm[..., i]  # Slice

        dcm.PhotometricInterpretation = "MONOCHROME2"
        dcm.SamplesPerPixel = 1
        dcm.BitsAllocated = 16
        dcm.BitsStored = 16
        dcm.HighBit = 15
        dcm.is_little_endian = True

        # Change default window levels for visualization
        dcm.fix_meta_info()
        dcm["PixelData"].is_undefined_length = False
        dcm.PixelData = s.tobytes()
        dcm.WindowCenter = s.max() // 2
        dcm.WindowWidth = s.max()
        path_save_dcm = path_save / f"{i}.dcm"
        pyd.dcmwrite(str(path_save_dcm), dcm)
