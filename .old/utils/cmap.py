import numpy as np


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

class ColourMap(object):

    def __init__(self, N=256, normalised=False, dataset='voc'):
        dtype = 'float32' if normalised else 'uint8'

        if dataset == 'voc' or dataset == 'nyu':
            self.map = np.zeros((N, 3), dtype=dtype)
            for i in range(N):
                r = g = b = 0
                c = i
                for j in range(8):
                    r = r | (bitget(c, 0) << 7 - j)
                    g = g | (bitget(c, 1) << 7 - j)
                    b = b | (bitget(c, 2) << 7 - j)
                    c = c >> 3

                self.map[i] = np.array([r, g, b])
        else:
            self.map = np.array([
                                [128,64,128],
                                [244,35,232],
                                [ 70,70,70],
                                [102,102,156],
                                [190,153,153],
                                [153,153,153],
                                [250,170,30],
                                [220,220,0],
                                [107,142,35],
                                [152,251,152],
                                [ 70,130,180],
                                [220,20,60],
                                [255,0,0],
                                [0,0,142],
                                [0,0,70],
                                [0,60,100],
                                [0,80,100],
                                [0,0,230],
                                [119,11,32]
                                ], dtype=dtype)

        self.map = self.map / 255 if normalised else self.map

    def colourise(self, prediction):
        prediction = self.map[prediction]
        return prediction
