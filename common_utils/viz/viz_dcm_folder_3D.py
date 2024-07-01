from pathlib import Path

import sass
from common_utils import data_loader, preprocessor


def main(path_dcm_folder: str):
    path_dcm_folder = Path(path_dcm_folder)
    npy = data_loader.load_data(
        path_data=path_dcm_folder, data_format="dicom", normalize=True
    )

    sass.scroll(npy)
    npy = preprocessor.resize_vol(npy, size=256)
    npy = preprocessor.crop_fov(npy, target_fov=240)
    sass.scroll(npy)


if __name__ == "__main__":
    path_dcm_folder = r"D:\Sravan\Data\Source\ADNI-T2star\006_S_4485"
    main(path_dcm_folder)
