# creates the mappings COCO to VOC classes
def coco2voc():
    voc2coco = [[0],
                [5],
                [2],
                [16],
                [9],
                [44],
                [6],
                [3, 8],
                [17],
                [62],
                [21],
                [67],
                [18],
                [19],
                [4],
                [1],
                [64],
                [20],
                [63],
                [7],
                [72]]
    coco2voc = [0] * 91
    for voc_idx in range(len(voc2coco)):
        for coco_idx in voc2coco[voc_idx]:
            coco2voc[coco_idx] = voc_idx

    return coco2voc