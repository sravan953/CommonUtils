from pathlib import Path

import sass
from common_utils import data_loader


def main(path_read_dcm1: Path, path_read_dcm2: Path):
    vol1 = data_loader.load_data(
        path_data=path_read_dcm1, data_format="dicom", normalize=True
    )
    vol2 = data_loader.load_data(
        path_data=path_read_dcm2, data_format="dicom", normalize=True
    )

    diff = vol1 - vol2
    sass.scroll(vol1, vol2, diff)


if __name__ == "__main__":
    path_read_dcm1 = Path(r"E:\Data\Source\AMRI-IP\S5\T2star\Fastest\s14275")
    path_read_dcm2 = Path(r"E:\Data\Source\AMRI-IP\S5\T2star\Denoised baseline\s14275"
                          )
    main(path_read_dcm1, path_read_dcm2)
