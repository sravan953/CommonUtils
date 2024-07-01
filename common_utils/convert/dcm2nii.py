from pathlib import Path
from typing import List, Union

import dicom2nifti as d2n


def __get_all_dicom_folders(path_read_dicom: Path) -> Union[Path, List[Path]]:
    print('Gathering all DICOM directories...')

    files = list(path_read_dicom.glob('**/*'))

    if len(files) == 0:  # path_read_dicom is a DICOM file itself
        print('Found 1')
        return path_read_dicom.parent
    else:  # path_read_dicom could be any nested strucutre of DICOM files
        folders = []
        for f in files:
            if f.is_file():
                if f.parent not in folders:
                    folders.append(f.parent)
        print(f'Found {len(folders)}')
        return folders


def __prepare_save_dir(path_read_dicom: Path, path_dcm_folder: Path, path_save_nii: Path) -> Path:
    if not path_save_nii.is_absolute():  # path_save_nii is relative, create this inside path_read_dicom
        path_save_nii = path_read_dicom / path_save_nii
    else:  # path_save_nii is absolute
        path_save_nii = path_save_nii / path_dcm_folder.relative_to(path_read_dicom)  # Preserve sub-folder structure

    if not path_save_nii.exists():
        path_save_nii.mkdir(parents=True, exist_ok=True)

    return path_save_nii


def dcm2nii(path_read_dicom: Path, path_save_nii: Path):
    """
    Determine if path_read_dicom is:
    1. DICOM directory
    2. Folder of DICOM directories
    3. DICOM file
    """
    dicom_folders = __get_all_dicom_folders(path_read_dicom)  # Get all DICOM folders at/inside this path
    for i, dcm_folder in enumerate(dicom_folders):  # Iterate
        print(f'{i + 1}/{len(dicom_folders)}')

        save_dcm_folder = __prepare_save_dir(path_read_dicom, dcm_folder, path_save_nii)  # Construct save path
        d2n.convert_directory(dicom_directory=dcm_folder, output_folder=save_dcm_folder, compression=False,
                              reorient=True)


if __name__ == '__main__':
    path_read_dicom = Path(r"D:\Sravan\Data\Source\AMRI-IP\S5\DWI\b0")
    path_save_nii = Path(r"D:\Sravan\Data\Source\AMRI-IP\S5\DWI\b0 nifti")

    dcm2nii(path_read_dicom, path_save_nii)
