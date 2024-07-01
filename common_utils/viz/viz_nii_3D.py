from pathlib import Path

import sass
from common_utils import data_loader


def main(path_nii: Path, dataset: str):
    if path_nii.is_dir():
        files = data_loader.glob_nifti(path_nii)
        for f in files:
            nii = data_loader.load_data(
                path_data=f,
                data_format="nifti",
                normalize=True,
                nifti_dataset=dataset,
                central_50pc_crop=False,
            )
            print(nii.shape)
            sass.scroll(nii)
    else:
        nii = data_loader.load_data(
            path_data=path_nii,
            data_format="nifti",
            normalize=True,
            nifti_dataset=dataset,
            central_50pc_crop=False,
        )
        print(nii.shape)
        sass.scroll(nii)


if __name__ == "__main__":
    path_nii = Path(r"D:\Sravan\Data\Source\ArtifactID\HCP-T1\mgh_1001\MPRAGE_GradWarped_and_Defaced\2013-01-01_11_25_56.0\S227198\HCP_mgh_1001_MR_MPRAGE_GradWarped_and_Defaced_Br_20140919084711597_S227198_I444246.nii")
    dataset = "HCP-T1"
    main(path_nii, dataset)
