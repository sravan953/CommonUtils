from pathlib import Path

import sass
from common_utils import data_loader


def main(path_dcm_folder: str):
    path_dcm_folder = Path(path_dcm_folder)
    npy = data_loader.load_data(
        path_data=path_dcm_folder, data_format="dicom", normalize=True
    )

    sass.scroll(npy)


if __name__ == "__main__":
    path_dcm_folder = r"D:\Sravan\Data\Source\ADNI-T2star\006_S_4485"
    main(path_dcm_folder)
