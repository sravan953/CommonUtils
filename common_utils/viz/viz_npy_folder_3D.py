from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

import sass
from common_utils import preprocessor


def main(path_read_npy: Path, viz_3D=False):
    files = list(path_read_npy.glob('**/*.npy'))
    arr = []
    for f in files:
        npy = np.load(str(f))
        npy = npy.astype(np.float).squeeze()
        npy = cv2.resize(npy, (256, 256))  # Resize
        npy = preprocessor.normalize_per_slice(npy)  # Normalize

        if viz_3D:
            arr.append(npy)
        else:
            plt.imshow(npy, cmap='gray')
            plt.show()

    if viz_3D:
        arr = np.stack(arr, axis=-1)
        sass.scroll(arr)


if __name__ == '__main__':
    path_read_npy = Path(
        r"D:\Sravan\Data\Datagen\ArtifactID_v2\IXI-T1\gibbs\1")
    main(path_read_npy, viz_3D=False)
