from pathlib import Path

import nibabel as nb
import numpy as np


def save_nii(npy: np.ndarray, path_save_nii: Path):
    npy = npy.astype(np.int16)
    nii = nb.Nifti1Image(npy, affine=np.eye(4))
    nb.save(img=nii, filename=str(path_save_nii))
