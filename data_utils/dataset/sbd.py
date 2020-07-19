import os
import numpy as np
import torch
import random
from PIL import Image
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset

# read list of files corresponding to train/val/test split
def read_filelist(file):
    file_list = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            file_list.append(line.strip())
            line = f.readline()

    return file_list

# Custom dataset class for Pascal VOC
class SBD(Dataset):
    '''Semantic Boundaries Segmentation dataset.'''

    def __init__(self, root_dir, image_set='train', transform=None, target_transform=None):
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
            self.file_list = read_filelist(os.path.join(root_dir, 'benchmark_RELEASE', 'dataset', 'train.txt'))
        elif self.image_set == 'val':
            self.file_list = read_filelist(os.path.join(root_dir, 'benchmark_RELEASE', 'dataset', 'val.txt'))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get filename
        filename = self.file_list[idx]

        # load image data
        img_name = os.path.join(self.root_dir, 'benchmark_RELEASE', 'dataset', 'img', filename + '.jpg')
        image = Image.open(img_name)
        image = image.convert('RGB')

        # load label data
        label_name = os.path.join(self.root_dir, 'benchmark_RELEASE', 'dataset', 'cls', filename + '.mat')
        label = loadmat(label_name)
        label = label['GTcls']['Segmentation'][0][0]
        label = Image.fromarray(label)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.transform:
            image = self.transform(image)

        random.seed(seed)
        if self.target_transform:
            label = self.target_transform(label)

        # convert to label to tensor (without scaling to [0,1])
        label = np.asarray(label).astype(np.int64)
        label = torch.from_numpy(label).type(torch.LongTensor)

        # create sample of data and label
        sample = {'name': filename, 'data': image, 'label': label}

        return sample
