import numpy as np


def __get_DCM_filename(filename):
    try:
        return int(filename.stem)
    except:
        return filename


def __get_MRDC_num(filename):
    n = str(filename).split('.')[-1]
    return int(n)


def sort_DCM_filenames(files: list) -> list:
    first_file = files[0]
    is_MRDC = "MRDC" in first_file.name
    if is_MRDC:
        return sorted(files, key=__get_MRDC_num)
    else:
        return sorted(files, key=__get_DCM_filename)


def sort_DCM_filenames_special(files: list) -> list:
    # Sort Yasmine Hurd's data
    files_sorted = sorted(files, key=lambda filename: int(filename.stem.split('-')[-2]))
    if all(np.array(files) == np.array(files_sorted)):
        files_sorted = sorted(files, key=lambda filename: int(filename.stem.split('.')[-4]))

    return files_sorted
