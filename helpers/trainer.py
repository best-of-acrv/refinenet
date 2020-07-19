import timeit
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.miou import compute_cm, compute_iu

class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()

        # create directory for outputs
        self.max_epochs = args.max_epochs
        self.eval_interval = args.eval_interval
        self.display_interval = args.display_interval
        self.snapshot_interval = args.snapshot_interval
        self.load_directory = args.load_directory
        self.save_directory = args.save_directory

    def train(self, args, model, dataset, val_dataset=None):

        # setup dataloaders
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        if val_dataset:
            val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        # define loss criterion
        criterion = nn.NLLLoss(ignore_index=dataset.ignore_index)

        # make save directory
        os.makedirs(self.save_directory, exist_ok=True)

        # training
        model.train()
        start = timeit.default_timer()
        curr_iteration = 0
        for epoch in range(self.max_epochs):
            for batch in dataloader:

                # extract data and labels from batch
                data = batch['data']
                labels = batch['label']
                if model.cuda_available:
                    data = data.cuda()
                    labels = labels.cuda()

                # fit data to model
                mean_loss = model.fit(data, labels, criterion)

                # display current training loss
                if (curr_iteration + 1) % self.display_interval == 0:
                    stop = timeit.default_timer()
                    print('[Epoch: {}] [Iter: {}] [loss: {:4f}]'.format(epoch, curr_iteration + 1, mean_loss.item()))
                    start = stop


                # evaluate model by computing mean IU
                if (curr_iteration + 1) % self.eval_interval == 0 and val_dataset:

                    # set model to eval first
                    model.eval()
                    mean_iu = model.validate(val_dataset)

                    print('[Epoch: {}] [Iter: {}] [mean IU: {:4f}]'.format(epoch, curr_iteration + 1, mean_iu))
                    with open(os.path.join(self.save_directory, 'mean_iu.txt'), 'a') as f:
                        f.writelines(['Iter: ', str(curr_iteration + 1), ' ', 'Mean IU: ', str(mean_iu), '\n'])

                    # model back to train mode
                    model.train()

                # save the model
                if (curr_iteration + 1) % self.snapshot_interval == 0:
                    model.save(curr_iteration + 1, self.save_directory)

                curr_iteration += 1

