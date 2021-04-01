# creates the mappings COCO to VOC classes
def coco2voc():
    voc2coco = [[0], [5], [2], [16], [9], [44], [6], [3, 8], [17], [62], [21],
                [67], [18], [19], [4], [1], [64], [20], [63], [7], [72]]
    coco2voc = [0] * 91
    for voc_idx in range(len(voc2coco)):
        for coco_idx in voc2coco[voc_idx]:
            coco2voc[coco_idx] = voc_idx

    return coco2voc


# creates the mappings of full Citiscapes classes to training classes
def cs2train():
    train2cs = [[7], [8], [11], [12], [13], [17], [19], [20], [21], [22], [23],
                [24], [25], [26], [27], [28], [31], [32], [33]]
    cs2train = [255] * 34
    for train_idx in range(len(train2cs)):
        for cs_idx in train2cs[train_idx]:
            cs2train[cs_idx] = train_idx

    return cs2train


# read list of files corresponding to train/val/test split
def read_filelist(file):
    file_list = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            file_list.append(line.strip())
            line = f.readline()

    return file_list
