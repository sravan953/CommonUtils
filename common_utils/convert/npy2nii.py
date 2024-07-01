from pathlib import Path

import nibabel as nb
import numpy as np


def main(path_save_nii: Path, npy: np.ndarray = np.array([]), path_read_npy: Path = Path()):
    """
    Save `npy` as a NIFTI at `path_save_nii`, or load from `path_read_npy` and save it as a NIFTI at `path_save_nii`.

    """
    if path_read_npy != Path() and npy == np.array([]):  # Load numpy
        npy = np.load(str(path_read_npy))
    elif path_read_npy == Path() and npy == np.array([]):
        raise ValueError('Either npy or path_read_npy must be passed.')
    elif path_read_npy != Path() and npy != np.array([]):
        raise ValueError('Either npy or path_read_npy must be passed, not both.')

    # Save as NIFTI
    nii = nb.Nifti1Image(npy, affine=np.eye(4))
    nb.save(img=nii, filename=str(path_save_nii))


if __name__ == '__main__':
    path_read_npy = Path(r"")
    path_save_nii = Path(r"")
    main(path_read_npy=path_read_npy, path_save_nii=path_save_nii)
