import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


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
            prediction = (_predict_multi_scale if self.multi_scale else
                          _predict_single_scale)(model, img)

            # convert PyTorch tensor to ndarray
            prediction = prediction.cpu().detach().numpy().astype(np.uint8)

            # Return prediction with colour map applied
            return colour_map.colourise(prediction + label_offset)


def _predict_multi_scale(model, img):

    # sample multi scale images
    height = img.shape[-2]
    width = img.shape[-1]

    # scales
    scales = [0.4, 0.6, 0.8, 1, 1.2]

    # predict at multiple scales
    predictions = []
    for scale in scales:

        # compute scaled height and width
        scaled_height = int(scale * height)
        scaled_width = int(scale * width)

        # interpolate image to scale
        scaled_img = F.interpolate(img, (scaled_height, scaled_width),
                                   mode='bilinear')

        # forward pass through
        logits = model(scaled_img)

        # interpolate logits back to original image size
        prediction = F.softmax(logits, dim=1)
        prediction = F.interpolate(prediction, (height, width),
                                   mode='bilinear')
        predictions.append(prediction)

    prediction = torch.cat(predictions, dim=0)
    prediction = torch.mean(prediction, dim=0)

    # average across images
    prediction = torch.argmax(prediction, dim=0)
    prediction = torch.squeeze(prediction)

    return prediction


def _predict_single_scale(model, img):

    # forward pass through
    logits = model(img)

    # interpolate logits back to original image size
    prediction = F.softmax(logits, dim=1)
    prediction = F.interpolate(prediction, (img.shape[-2], img.shape[-1]),
                               mode='bilinear')
    prediction = torch.argmax(prediction, dim=1)
    prediction = torch.squeeze(prediction)

    return prediction
