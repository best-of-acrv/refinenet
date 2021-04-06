import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data.dataset import Dataset

from .helpers import read_filelist
from ..helpers import ColourMap


class NYU(Dataset):
    '''NYUv2-40 Segmentation dataset.'''
    COLOUR_MAP = ColourMap(dataset='voc')
    LABEL_OFFSET = 1
    NUM_CLASSES = 40

    def __init__(self,
                 root_dir,
                 image_set='train',
                 transform=None,
                 target_transform=None):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''

        self.root_dir = root_dir
        self.image_set = image_set

        if self.image_set == 'train':
            self.file_list = read_filelist(os.path.join(root_dir, 'train.txt'))
        elif self.image_set == 'test':
            self.file_list = read_filelist(os.path.join(root_dir, 'test.txt'))
        self.transform = transform
        self.target_transform = target_transform

        # dataset properties
        self.num_classes = NYU.NUM_CLASSES
        self.ignore_index = 255
        self.label_offset = NYU.LABEL_OFFSET
        self.cmap = NYU.COLOUR_MAP

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get filename
        filename = self.file_list[idx]

        # load image data
        img_name = os.path.join(self.root_dir, 'images',
                                'img_' + filename + '.png')
        image = Image.open(img_name)
        image = image.convert('RGB')

        # load label data
        label_name = os.path.join(self.root_dir, 'labels',
                                  'gt_' + filename + '.png')
        label = Image.open(label_name)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.transform:
            image = self.transform(image)

        random.seed(seed)
        if self.target_transform:
            label = self.target_transform(label)

        # convert to label to tensor (without scaling to [0,1])
        label = np.asarray(label).astype(np.uint8)
        # shift labels by -1 to remove void
        label = label - 1
        label = torch.from_numpy(label).type(torch.LongTensor)

        # create sample of data and label
        sample = {'name': filename, 'data': image, 'label': label}

        return sample
