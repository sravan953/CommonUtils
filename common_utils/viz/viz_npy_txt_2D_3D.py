from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import sass


def main(path_read_txt: Path, viz_3D=False):
    text = path_read_txt.read_text().splitlines()[1:]
    arr = []
    for line in text:
        npy = np.load(str(path_read_txt.parent / line.strip()))
        npy = npy.astype(np.float).squeeze()

        if viz_3D:
            arr.append(npy)
        else:
            plt.imshow(npy, cmap='gray')
            plt.axis('off')
            plt.show()

    if viz_3D:
        arr = np.stack(arr, axis=-1)
        sass.scroll(arr)


if __name__ == '__main__':
    path_read_txt = Path(r"D:\Sravan\Data\Datagen\ArtifactID\IXI-T1\valv2.txt")
    main(path_read_txt, viz_3D=True)
