import os
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from data_utils.mappings import cs2train
from utils.cmap import ColourMap

# generate list of image/label filepaths
def generate_image_list(data_dir, label_dir):
    image_subdirs = next(os.walk(data_dir))[1]
    label_subdirs = next(os.walk(label_dir))[1]

    # create image file list
    img_files = []
    for subdir in image_subdirs:
        curr_dir = os.path.join(data_dir, subdir)
        img_files += [os.path.join(curr_dir, k) for k in os.listdir(curr_dir)]

    # create image file list
    label_files = []
    for subdir in label_subdirs:
        curr_dir = os.path.join(label_dir, subdir)
        label_files += [os.path.join(curr_dir, k) for k in os.listdir(curr_dir) if 'labelIds' in k]

    # sort image and label files
    img_files.sort()
    label_files.sort()

    # verify image and label files are aligned
    for pair in zip(img_files, label_files):
        img_name = pair[0]
        label_name = pair[1]
        img_name = os.path.basename(img_name)
        label_name = os.path.basename(label_name)
        img_name = img_name.split('_')
        label_name = label_name.split('_')

        assert img_name[0] == label_name[0]
        assert img_name[1] == label_name[1]
        assert img_name[2] == label_name[2]

    return img_files, label_files


class Citiscapes(Dataset):
    '''Citiscapes dataset.'''

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
        self.cs2train = np.asarray(cs2train())

        if self.image_set == 'train':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'train')
            self.label_dir = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'train')
        elif self.image_set == 'val':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'val')
            self.label_dir = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'val')
        elif self.image_set == 'test':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'test')
            self.label_dir = os.path.join(root_dir, 'gtFine_trainvaltest', 'gtFine', 'test')
        elif self.image_set == 'train_extra':
            self.image_dir = os.path.join(root_dir, 'leftImg8bit_trainextra', 'leftImg8bit', 'train_extra')
            self.label_dir = os.path.join(root_dir, 'gtCoarse', 'gtCoarse', 'train_extra')

        # generate file list for training and label images
        self.image_list, self.label_list = generate_image_list(self.image_dir, self.label_dir)

        self.transform = transform
        self.target_transform = target_transform

        # dataset properties
        self.num_classes = 19
        self.ignore_index = 255
        self.label_offset = 0
        self.cmap = ColourMap(dataset='cs')


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get filename
        img_filepath = self.image_list[idx]
        label_filepath = self.label_list[idx]

        # base filename
        filename = os.path.basename(img_filepath)
        filename = filename.split('_')
        filename = filename[0] + '_' + filename[1] + '_' + filename[2]

        # load image data
        image = Image.open(img_filepath)
        image = image.convert('RGB')

        # load label data
        label = Image.open(label_filepath)

        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.transform:
            image = self.transform(image)

        random.seed(seed)
        if self.target_transform:
            label = self.target_transform(label)

        # convert to label to tensor (without scaling to [0,1])
        label = np.asarray(label).astype(np.uint8)

        # convert full set of citiscapes labels to training labels
        label = self.cs2train[label]

        # convert to tensor
        label = torch.from_numpy(label).type(torch.LongTensor)

        # create sample of data and label
        sample = {'name': filename, 'data': image, 'label': label}

        return sample
