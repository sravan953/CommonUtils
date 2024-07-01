from pathlib import Path
from typing import Union, List, Tuple

import nibabel as nb
import numpy as np
import pydicom as pyd

from common_utils import data_utils
from common_utils import preprocessor
from common_utils.sort_DCM import sort_DCM_filenames


def glob_dicom(path_dicom: Path) -> list:
    # .MRDC.* + .dcm
    dicom_files = list(path_dicom.glob("**/*.MRDC.*")) + list(
        path_dicom.glob("**/*.dcm")
    )
    return dicom_files


def glob_nifti(path_nifti: Path) -> list:
    nifti_files = list(path_nifti.glob("**/*.nii")) + list(
        path_nifti.glob("**/*.nii.gz")
    )
    return nifti_files


def load_dicom_folder(
        path_dicom_folder: Path, return_dicoms: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if any(
            (
                    "MRDC" in path_dicom_folder.stem,
                    "IMA" in path_dicom_folder.stem,
            )
    ):
        # path_dicom_folder is a DICOM file
        # It should be parent folder instead
        path_dicom_folder = path_dicom_folder.parent

    dicom_files = glob_dicom(path_dicom_folder)
    # *.MRDC.* DICOM FILES ARE NEVER IN ALPHABETICAL ORDER!!!
    dicom_files = sort_DCM_filenames(dicom_files)

    vol = []
    dicoms = []
    for d in dicom_files:
        dicom = pyd.dcmread(str(d))
        vol.append(dicom.pixel_array)

        if return_dicoms:
            dicoms.append(dicom)

    vol = np.stack(vol, axis=-1).astype(np.float)

    if return_dicoms:
        return vol, dicoms
    return vol


def load_nifti(path_nifti: Path, nifti_dataset: str) -> np.ndarray:
    nii = nb.load(str(path_nifti))
    vol = nii.get_data().squeeze()

    # Dataset-contrast specific preprocessing
    if nifti_dataset == "HCP-T2":
        vol = np.moveaxis(vol, 2, 1)
    elif nifti_dataset == "HCP-T1":
        vol = np.rot90(vol, -1)
        vol = np.fliplr(vol)
    elif nifti_dataset in ["IXI-T1", "IXI_3T-T1", "IXI_15T-T1"]:
        vol = np.rot90(vol, 1)
    elif nifti_dataset == "HCP-T1-BET":
        vol = np.rot90(vol, -1, axes=(0, 1))
        vol = np.fliplr(vol)
    elif nifti_dataset == "IXI-T1-BET":
        vol = np.rot90(vol, 1, axes=(0, 1))
    elif nifti_dataset == "HCP-T1-FLIRT":
        vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
        vol = np.fliplr(vol)
        vol = np.flipud(vol)
        vol = np.pad(vol, ((37, 37), (19, 19), (0, 0)))
    elif nifti_dataset == "IXI-T1-FLIRT":
        vol = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
        vol = np.fliplr(vol)
        vol = np.flipud(vol)
        vol = np.pad(vol, ((37, 37), (19, 19), (0, 0)))
    elif nifti_dataset == "HCP-T1-FLIRT-BET":
        vol = np.moveaxis(vol, 0, 2)
        vol = np.rot90(vol, axes=(0, 1))
        vol = np.fliplr(vol)
        vol = np.pad(vol, ((37, 37), (19, 19), (0, 0)))
    elif nifti_dataset == "IXI-T1-FLIRT-BET":
        vol = np.moveaxis(vol, 0, 2)
        vol = np.rot90(vol, axes=(0, 1))
        vol = np.fliplr(vol)
        vol = np.pad(vol, ((37, 37), (19, 19), (0, 0)))
    elif nifti_dataset == "MS_SEG-FLAIR":
        vol = np.rot90(vol, 1)
        vol = preprocessor.resize_vol(vol, 256)
    elif nifti_dataset == "ADNI_GO2-T1-FLIRT-BET":
        vol = np.moveaxis(vol, 0, 2)
        vol = np.rot90(vol, axes=(0, 1))
        vol = np.fliplr(vol)
        vol = np.pad(vol, ((37, 37), (19, 19), (0, 0)))
    elif nifti_dataset == "ADNI-T2star":
        vol = np.rot90(vol, 1)

    vol = vol.astype(np.float)  # Convert from np.memmap

    return vol


def load_numpy(path_numpy: Path) -> np.ndarray:
    if path_numpy.is_dir():
        # Load folder as a numpy volume
        npy = []
        for f in path_numpy.glob("*.npy"):
            each_npy = np.load(str(f))
            npy.append(each_npy)
        npy = np.stack(npy, axis=-1)
        return npy
    return np.load(str(path_numpy))  # Load single numpy


def load_numpy_from_list(files: list) -> np.ndarray:
    npy = []
    for f in files:
        npy_noisy = np.load(str(f))
        npy.append(npy_noisy)
    npy = np.stack(npy, axis=-1)

    return npy


def load_data(
        path_data: Union[Path, list],
        data_format: str,
        normalize: bool,
        central_50pc_crop: bool = False,
        nifti_dataset: str = "",
        return_dicoms: bool = False,
        target_size: int = None,
):
    if not isinstance(path_data, (Path, list)):
        path_data = Path(path_data)

    if data_format == "nifti":  # Load NIFTI
        data = load_nifti(path_data, nifti_dataset=nifti_dataset)
    elif data_format == "dicom":  # Load DICOM
        data = load_dicom_folder(path_data, return_dicoms)
        if return_dicoms:  # Return DICOM files also
            data, dicoms = data
    elif data_format in ("npy", "numpy"):  # Load npy
        data = load_numpy(path_data)
    elif data_format == "list":
        data = load_numpy_from_list(path_data)
    else:
        raise ValueError("Unknown data format. Expected nifti, dicom or npy")
    data = data.squeeze()

    if target_size is not None:  # Resize to target_size
        data = preprocessor.resize_vol(data, target_size)

    if normalize:  # Normalize data
        data = preprocessor.normalize_volume(data)

    if central_50pc_crop:
        data = data_utils.crop_central_50pc(data)

    # # Debug - visualize
    # print(f"Debug - visualize {path_data}")
    # import sass
    #
    # sass.scroll(data)

    if return_dicoms:
        return data, dicoms
    return data
