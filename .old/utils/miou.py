import torch
import numpy as np

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

    cm = torch.zeros([N*N], dtype=torch.float32)
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



