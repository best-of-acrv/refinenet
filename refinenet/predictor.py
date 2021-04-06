import os
import torch
import torch.nn as nn
import numpy as np

from .helpers import forward_multi_scale, forward_single_scale


class Predictor(nn.Module):

    def __init__(self, multi_scale=False):
        super().__init__()

        self.multi_scale = multi_scale

    # sample images using specified snapshot model
    def predict(self, image, model, colour_map, label_offset):
        with torch.no_grad():
            # move image to GPU
            img = image.cuda() if model.cuda_available else image

            # turn single image into NCHW format
            if len(img.shape) < 4:
                img = torch.unsqueeze(img, 0)

            # predict using single or multi-scale images
            prediction = (forward_multi_scale if self.multi_scale else
                          forward_single_scale)(model, img)

            # convert PyTorch tensor to ndarray
            prediction = prediction.cpu().detach().numpy().astype(np.uint8)

            # Return prediction with colour map applied
            return colour_map.colourise(prediction + label_offset)
