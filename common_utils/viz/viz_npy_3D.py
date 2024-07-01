from pathlib import Path

import sass
from common_utils import data_loader


def main(path_npy: Path):
    npy = data_loader.load_data(path_data=path_npy, data_format='npy', normalize=True)
    print(npy.max(), npy.min())
    sass.scroll(npy)


if __name__ == '__main__':
    path_npy = Path(r"D:\Sravan\Data\Source\202312-T1w_MtSinai_YasminHurd\datagen\gibbs")
    path_npy = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\noartifact\35")
    main(path_npy)
