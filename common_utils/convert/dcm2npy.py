from pathlib import Path

import numpy as np
import pydicom

from common_utils import preprocessor
from common_utils.sort_DCM import sort_DCM_filenames


def dcm2npy(path_read_dicom: Path, path_save_npy: Path, normalize: bool = True):
    dcm_files = list(path_read_dicom.glob("*.MRDC.*"))
    if len(dcm_files) == 0:
        dcm_files = list(path_read_dicom.glob("*.IMA"))
    else:
        dcm_files = sort_DCM_filenames(dcm_files)  # Sort DICOMs

    if not path_save_npy.exists():
        path_save_npy.mkdir(parents=False)

    # Convert individual DICOM files into a 3D Numpy vol
    vol = []
    for d in dcm_files:
        dcm = pydicom.dcmread(str(d))
        dcm = dcm.pixel_array

        vol.append(dcm)
    vol = np.stack(vol, axis=-1)

    if normalize:
        vol = preprocessor.normalize_volume(vol)

    for i, d in enumerate(dcm_files):
        path_save = path_save_npy / (d.stem + ".npy")
        np.save(arr=vol[..., i], file=str(path_save))


if __name__ == "__main__":
    path_read_dicom = Path(
        r"D:\CU Data\Source\ArtifactID\artifactID_Siemens\20211103_KSR\T1_MPRAGE_(SEGMENT_1)_MOTION_0006"
    )
    path_save_npy = Path(
        r"D:\CU Data\Source\ArtifactID\artifactID_Siemens\20211103_KSR\T1_MPRAGE_(SEGMENT_1)_MOTION_0006_npy"
    )

    dcm2npy(path_read_dicom, path_save_npy)
