import numpy as np
import torch


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def compute_cm(gt, pred, N, with_cuda=False):
    '''Computes confusion matrix between prediction and gt
    Args:
      gt (Tensor) : gt.
      pred (Tensor) : predictions.
      N (int) : number of classes in matrix.
    Returns:
      Confusion matrix
      (Tensor of size (N, N)).
    '''
    # flatten gt and preds
    pred = pred.view([-1])
    gt = gt.view([-1])

    # check for valid labels
    mask_1 = gt >= 0
    mask_2 = gt < N
    mask = torch.logical_and(mask_1, mask_2)

    # compute scatter idx
    pred = pred[mask]
    gt = gt[mask]
    gt *= N
    idx = gt + pred

    cm = torch.zeros([N * N], dtype=torch.float32)
    if with_cuda:
        cm = cm.cuda()
    samples = torch.ones_like(gt, dtype=torch.float)

    # compute confusion matrix and reshape
    cm = cm.scatter_add_(0, idx, samples)
    cm = cm.view([N, N]).type(torch.int64)

    return cm


def compute_iu(cm):
    '''Compute IU from confusion matrix.
    Args:
      cm (ndarray) : square confusion matrix.
    Returns:
      IU vector (ndarray).
    '''
    N = cm.shape[0]
    IU = np.ones(N)

    for i in range(N):
        fn = np.sum(cm[:, i])
        fp = np.sum(cm[i, :])
        tp = cm[i, i]
        total = fn + fp - tp
        if total > 0:
            IU[i] = tp / total
    return IU


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
            self.map = np.array(
                [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170, 30], [
                     220, 220, 0
                 ], [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]],
                dtype=dtype)

        self.map = self.map / 255 if normalised else self.map

    def colourise(self, prediction):
        prediction = self.map[prediction]
        return prediction
