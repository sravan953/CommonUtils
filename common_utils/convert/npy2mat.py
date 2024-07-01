from pathlib import Path

import numpy as np
from scipy.io import savemat


def main(path_read_npy: Path, path_save_mat: Path):
    # Load numpy
    npy = np.load(str(path_read_npy))

    # Save as .mat
    key = path_save_mat.stem
    savemat(file_name=str(path_save_mat), mdict={key: npy})


if __name__ == '__main__':
    path_read_npy2 = Path(r"C:\Users\sravan953\Desktop\Outputs")
    for path_read_npy in path_read_npy2.glob('**/*.npy'):
        path_save_mat = path_read_npy.with_suffix('.mat')
        main(path_read_npy=path_read_npy, path_save_mat=path_save_mat)
