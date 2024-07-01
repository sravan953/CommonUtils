from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import sass


def main(path_read_npy1: Path,
         path_read_npy2: Path,
         viz_3D=False):
    files1 = list(path_read_npy1.glob('**/*.npy'))
    files2 = list(path_read_npy2.glob('**/*.npy'))

    arr1 = []
    arr2 = []
    for i in range(len(files1)):
        npy1 = np.load(str(files1[i]))
        npy1 = npy1.astype(np.float).squeeze()

        npy2 = np.load(str(files2[i]))
        npy2 = npy2.astype(np.float).squeeze()

        if viz_3D:
            arr1.append(npy1)
            arr2.append(npy2)
        else:
            ax = plt.subplot(131)
            plt.imshow(npy1, cmap='gray')
            plt.axis('off')
            plt.subplot(132, sharex=ax, sharey=ax)
            plt.imshow(npy2, cmap='gray')
            plt.axis('off')
            plt.subplot(133, sharex=ax, sharey=ax)
            plt.imshow(npy2 - npy1, cmap='gray')
            plt.axis('off')
            plt.show()

    if viz_3D:
        arr1 = np.stack(arr1, axis=-1)
        arr2 = np.stack(arr2, axis=-1)
        diff = arr1 - arr2
        sass.scroll(arr1, arr2, diff)


if __name__ == '__main__':
    path_read_npy1 = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\gibbs\1")
    path_read_npy2 = Path(r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\noartifact\1")

    main(path_read_npy1,
         path_read_npy2,
         viz_3D=False)
