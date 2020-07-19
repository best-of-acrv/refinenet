import numpy as np


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

class ColourMap(object):

    def __init__(self, N=256, normalised=False):
        dtype = 'float32' if normalised else 'uint8'
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

        self.map = self.map / 255 if normalised else self.map

    def colourise(self, prediction):
        prediction = self.map[prediction]
        return prediction
