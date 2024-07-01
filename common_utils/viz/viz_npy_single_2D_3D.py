from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import sass


def main(path_read_npy: Path, viz_3D=False):
    npy = np.load(str(path_read_npy))
    npy = npy.astype(np.float).squeeze()

    if viz_3D:
        sass.scroll(npy)
    else:
        plt.imshow(npy, cmap='gray')
        plt.show()


if __name__ == '__main__':
    path_read_npy = Path(r"D:\Sravan\Data\Datagen\AMRI-IP\ADNI-T2star\noisy\018_S_4400\I1044188_34.npy")
    main(path_read_npy, viz_3D=False)
