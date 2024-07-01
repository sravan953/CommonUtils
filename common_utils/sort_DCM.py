def __get_DCM_filename(filename):
    return int(filename.stem)


def __get_MRDC_num(filename):
    n = str(filename).split('.')[-1]
    return int(n)


def advanced_sort(files):
    try:
        sorted_files = sorted(files, key=lambda f: int(f.stem.split(".")[8].split("-")[-2]))
    except:
        sorted_files = sorted(files, key=lambda f: int(f.stem.split(".")[-4]))

    return sorted_files


def sort_DCM_filenames(files: list) -> list:
    first_file = files[0]
    is_MRDC = "MRDC" in first_file.name
    if is_MRDC:
        return sorted(files, key=__get_MRDC_num)
    else:
        return advanced_sort(files)
