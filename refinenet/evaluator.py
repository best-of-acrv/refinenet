import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch

from .helpers import (compute_cm, compute_iu, forward_multi_scale,
                      forward_single_scale)


class Evaluator(nn.Module):
    MIU_FILENAME = 'mean_iu.txt'

    def __init__(self,
                 multi_scale=False,
                 output_directory='.',
                 output_images=False):
        super().__init__()

        self.multi_scale = multi_scale
        self.output_directory = output_directory
        self.output_images = output_images

    # sample images using specified snapshot model
    def sample(self, model, dataset):
        # create dataloader using dataset
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1)

        # Create the output directory for our images if required
        output_directory = (os.path.join(self.output_directory, 'images')
                            if self.output_images else None)
        if output_directory is not None:
            os.makedirs(output_directory, exist_ok=True)

        # set model to eval mode for inference
        model.eval()
        # sample images and generate prediction images
        with torch.no_grad():
            for batch in dataloader:

                # retrieve required data from batch
                name = batch['name']
                img = batch['data']
                if model.cuda_available:
                    img = img.cuda()

                # predict using single or multi-scale images
                prediction = (forward_multi_scale if self.multi_scale else
                              forward_single_scale)(model, img)

                # convert PyTorch tensor to ndarray
                prediction = prediction.cpu().detach().numpy().astype(np.uint8)

                # apply colour map
                prediction += dataset.label_offset
                prediction = dataset.cmap.colourise(prediction)

                # save prediction image if requested
                if output_directory is not None:
                    Image.fromarray(prediction).save(
                        os.path.join(output_directory, '%s.png' % name[0]))

    def compute_miu(self, model, dataset):
        # create dataloader using dataset
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1)

        # full confusion matrix
        full_cm = torch.zeros((dataset.num_classes, dataset.num_classes),
                              dtype=torch.int64)
        if model.cuda_available:
            full_cm = full_cm.cuda()

        # set model to eval mode for inference
        model.eval()
        # sample images and generate prediction images
        with torch.no_grad():
            for batch in dataloader:

                # retrieve required data from batch
                img = batch['data']
                label = batch['label']
                if model.cuda_available:
                    img = img.cuda()
                    label = label.cuda()

                # predict using single or multi-scale images
                prediction = (forward_multi_scale if self.multi_scale else
                              forward_single_scale)(model, img)

                # compute IU
                cm = compute_cm(label, prediction, dataset.num_classes,
                                model.cuda_available)

                # add to total confusion matrix
                full_cm += cm

            # compute mean IU from confusion matrix
            full_cm = full_cm.cpu().detach().numpy().astype(np.int64)
            iu = compute_iu(full_cm)
            mean_iu = np.mean(iu)

            with open(
                    os.path.join(self.output_directory,
                                 Evaluator.MIU_FILENAME), 'a') as f:
                f.writelines(['Mean IU: ', str(mean_iu), '\n'])
