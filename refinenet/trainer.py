import timeit
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(nn.Module):

    def __init__(self, output_directory=None):
        super().__init__()

        # Declare directory for outputs
        self.output_directory = output_directory

    def train(self, model, dataset, *, eval_interval, snapshot_interval,
              display_interval, batch_size, num_workers, freeze_batch_normal):

        # freeze batch norm during training
        if freeze_batch_normal:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # make save directory
        if self.output_directory is not None:
            os.makedirs(self.output_directory, exist_ok=True)

        # go through each dataset
        curr_epoch = 0
        curr_iteration = 0
        for i, d in enumerate(dataset['train']):

            # create schedulers for encoder and decoder optimisers
            enc_scheduler = torch.optim.lr_scheduler.StepLR(
                model.enc_optimiser,
                step_size=dataset['stage_epochs'][i],
                gamma=dataset['stage_gammas'][i])
            dec_scheduler = torch.optim.lr_scheduler.StepLR(
                model.dec_optimiser,
                step_size=dataset['stage_epochs'][i],
                gamma=dataset['stage_gammas'][i])

            # setup dataloader
            dataloader = DataLoader(d,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers)

            # define loss criterion
            criterion = nn.NLLLoss(ignore_index=d.ignore_index)

            # training
            model.train()
            start = timeit.default_timer()
            for epoch in range(dataset['stage_epochs'][i]):

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
                    if (curr_iteration + 1) % display_interval == 0:
                        stop = timeit.default_timer()
                        print(
                            '[Epoch: {}] [Iter: {}] [Time: {:3f}] [Lr: {}] [loss: {:4f}]'
                            .format(curr_epoch, curr_iteration + 1,
                                    stop - start, get_lr(model.enc_optimiser),
                                    mean_loss.item()))
                        start = stop
                    curr_iteration += 1

                # evaluate model by computing mean IU
                if (epoch + 1) % eval_interval == 0 and dataset['val']:
                    # set model to eval first
                    model.eval()
                    mean_iu = model.validate(dataset['val'])
                    print('[Epoch: {}] [Iter: {}] [mean IU: {:4f}]'.format(
                        curr_epoch, curr_iteration + 1, mean_iu))
                    if self.output_directory is not None:
                        with open(
                                os.path.join(self.output_directory,
                                             'mean_iu.txt'), 'a') as f:
                            f.writelines([
                                'Epoch: ',
                                str(curr_epoch + 1), ' ', 'Mean IU: ',
                                str(mean_iu), '\n'
                            ])

                    # model back to train mode
                    model.train()

                # save the model
                if (epoch + 1) % snapshot_interval == 0:
                    model.save(curr_epoch + 1, self.output_directory)

                # step scheduler
                enc_scheduler.step()
                dec_scheduler.step()

                # update current epoch
                curr_epoch += 1
