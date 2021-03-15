# read list of files corresponding to train/val/test split
def read_filelist(file):
    file_list = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            file_list.append(line.strip())
            line = f.readline()

    return file_list
