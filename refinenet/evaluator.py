import os
import torch
import torch.nn as nn
import numpy as np
from utils.miou import compute_cm, compute_iu


class Evaluator(nn.Module):

    def __init__(self, args, multi_scale=False):
        super().__init__()

        self.sample_directory = args.save_directory
        self.multi_scale = multi_scale

    # sample images using specified snapshot model
    def sample(self, model, dataset):
        # create dataloader using dataset
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1)

        # output directory
        output_directory = os.path.join(self.sample_directory, 'output',
                                        model.name)
        os.makedirs(output_directory, exist_ok=True)

        # set model to eval mode for inference
        model.eval()
        # sample images and generate prediction images
        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # retrieve required data from batch
                name = batch['name']
                img = batch['data']
                if model.cuda_available:
                    img = img.cuda()

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
                output_file = os.path.join(output_directory, name[0] + '.png')
                im.save(output_file)

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
            for i, batch in enumerate(dataloader):

                # retrieve required data from batch
                img = batch['data']
                label = batch['label']
                if model.cuda_available:
                    img = img.cuda()
                    label = label.cuda()

                # predict using single or multi-scale images
                if self.multi_scale:
                    prediction = self.eval_multi_scale(model, img)
                else:
                    prediction = self.eval_single_scale(model, img)

                # compute IU
                cm = compute_cm(label, prediction, dataset.num_classes,
                                model.cuda_available)

                # add to total confusion matrix
                full_cm += cm

            # compute mean IU from confusion matrix
            full_cm = full_cm.cpu().detach().numpy().astype(np.int64)
            iu = compute_iu(full_cm)
            mean_iu = np.mean(iu)

            with open(os.path.join(self.sample_directory, 'mean_iu.txt'),
                      'a') as f:
