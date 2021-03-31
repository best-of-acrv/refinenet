import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from data_utils.transforms import get_transforms


class Predictor(nn.Module):
    def __init__(self, args, multi_scale=False):
        super().__init__()

        self.sample_directory = args.save_directory
        self.multi_scale = multi_scale

    # sample images using specified snapshot model
    def run_on_single_image(self, model, dataset, img_path):
        # output directory
        output_directory = os.path.join(self.sample_directory, 'output',
                                        model.name)
        os.makedirs(output_directory, exist_ok=True)

        # sample images and generate prediction images
        with torch.no_grad():
            # load image data
            img = Image.open(img_path)
            img = img.convert('RGB')

            # get image transform for evaluation
            transform_val, _ = get_transforms(mode='eval')

            # apply transforms for evaluation
            img = transform_val(img)

            # move image to GPU
            if model.cuda_available:
                img = img.cuda()

            # turn single image into NCHW format
            if len(img.shape) < 4:
                img = torch.unsqueeze(img, 0)

            # predict using single or multi-scale images
            if self.multi_scale:
                prediction = self.eval_multi_scale(model, img)
            else:
                prediction = self.eval_single_scale(model, img)

            # convert PyTorch tensor to ndarray
            prediction = prediction.cpu().detach().numpy().astype(np.uint8)

            # apply colour map
            prediction += dataset.label_offset
            prediction = dataset.cmap.colourise(prediction)

            # save prediction image
            im = Image.fromarray(prediction)
            output_file = os.path.join(output_directory,
                                       os.path.basename(img_path))
            im.save(output_file)

    def eval_single_scale(self, model, img):

        # forward pass through
        logits = model(img)

        # interpolate logits back to original image size
        prediction = F.softmax(logits, dim=1)
        prediction = F.interpolate(prediction, (img.shape[-2], img.shape[-1]),
                                   mode='bilinear')
        prediction = torch.argmax(prediction, dim=1)
        prediction = torch.squeeze(prediction)

        return prediction

    def eval_multi_scale(self, model, img):

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
