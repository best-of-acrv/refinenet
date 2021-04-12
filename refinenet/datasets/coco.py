import os
import pickle
import torch
import random
import numpy as np
from tqdm import trange
from PIL import Image
from pycocotools import coco
from pycocotools import mask as cocomask
from torch.utils.data import Dataset

from .helpers import coco2voc
from ..helpers import ColourMap


class COCO(Dataset):
    '''Microsoft COCO Segmentation dataset.'''
    COLOUR_MAP = ColourMap(dataset='voc')
    LABEL_OFFSET = 0
    NUM_CLASSES = 21

    def __init__(self,
                 root_dir,
                 image_set='train',
                 transform=None,
                 target_transform=None):

        self.root_dir = root_dir
        self.image_set = image_set
        if self.image_set == 'train':
            self.img_dir = os.path.join(root_dir, 'train2017')
            self.ann_file = os.path.join(
                root_dir, 'annotations/instances_train2017.json')
            self.ids_file = os.path.join(
                root_dir, 'annotations/instances_train2017_ids.mx')
        elif self.image_set == 'val':
            self.img_dir = os.path.join(root_dir, 'val2017')
            self.ann_file = os.path.join(root_dir,
                                         'annotations/instances_val2017.json')
            self.ids_file = os.path.join(
                root_dir, 'annotations/instances_val2017_ids.mx')
        self.transform = transform
        self.target_transform = target_transform

        # gets COCO to VOC classes mapping
        self.coco2voc = coco2voc()
        self.coco = coco.COCO(self.ann_file)
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.img_ids = self.coco.getImgIds()
        self.catIds = self.coco.getCatIds()

        # check if ids of qualified images exists, otherwise create list of qualified ids
        if os.path.exists(self.ids_file):
            with open(self.ids_file, 'rb') as f:
                self.img_ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.img_ids = self.preprocess(ids, self.ids_file)

        # dataset properties
        self.num_classes = COCO.NUM_CLASSES
        self.ignore_index = 255
        self.label_offset = COCO.LABEL_OFFSET
        self.cmap = COCO.COLOUR_MAP

    def __get_year(self):
        name = self.dataset_name
        if 'coco' in name:
            name = name.replace('coco', '')
        else:
            name = name.replace('COCO', '')
        year = name
        return year

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        # load image data
        img_ann = self.coco.loadImgs(self.img_ids[idx])

        # get filename
        filename = img_ann[0]['file_name']

        img_name = os.path.join(self.img_dir, filename)
        image = Image.open(img_name)
        image = image.convert('RGB')
        h, w = image.height, image.width

        # load label data
        ann_ids = self.coco.getAnnIds(imgIds=self.img_ids[idx])
        anns = self.coco.loadAnns(ann_ids)
        label = np.zeros((image.height, image.width), dtype=np.uint8)
        for ann_item in anns:
            mask = self.coco.annToMask(ann_item)
            label[mask > 0] = self.coco2voc[ann_item['category_id']]

        # check segmentation is correct & convert to PIL image for transform
        if np.max(label) > 91:
            print(np.max(label))
            raise ValueError('segmentation > 91')
        if np.max(label) > 20:
            print(np.max(label))
            raise ValueError('segmentation > 20')
        label = Image.fromarray(label)

        # ensure same random transformation being applied to both data and target
        seed = np.random.randint(2147483647)
        random.seed(seed)
        if self.transform:
            image = self.transform(image)

        random.seed(seed)
        if self.target_transform:
            label = self.target_transform(label)

        # convert to label to tensor (without scaling to [0,1])
        label = np.asarray(label).astype(np.uint8)
        label = torch.from_numpy(label).type(torch.LongTensor)

        # create sample of data and label
        sample = {'name': filename, 'data': image, 'label': label}

        return sample

    def gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for instance in target:
            rle = cocomask.frPyObjects(instance['segmentation'], h, w)
            m = cocomask.decode(rle)
            cat = instance['category_id']
            if self.coco2voc[cat] != 0:
                c = self.coco2voc[cat]
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * ((
                    (np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def preprocess(self, ids, ids_file):
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self.gen_seg_mask(cocotarget, img_metadata['height'],
                                     img_metadata['width'])
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description(
                'Doing: {}/{}, got {} qualified images'.format(
                    i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids
