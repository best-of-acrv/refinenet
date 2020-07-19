from torch.utils.data import ConcatDataset
from data_utils.dataset.voc import VOC
from data_utils.dataset.sbd import SBD
from data_utils.dataset.coco import COCODataset
from data_utils.dataset.nyu import NYU

def get_dataset(dataset, image_set='train', augment=False, transform=None, target_transform=None):
    if dataset == 'voc':
        # Pascal VOC Dataset
        voc_dataset = VOC(root_dir='data/pascal_voc',
                          image_set=image_set,
                          transform=transform,
                          target_transform=target_transform)
        if augment:
            # Semantic Boundaries Dataset
            sbd_dataset = SBD(root_dir='data/sbd',
                              image_set=image_set,
                              transform=transform,
                              target_transform=target_transform)
            # Microsoft COCO Dataset
            coco_dataset = COCODataset(root_dir='data/coco',
                                       image_set=image_set,
                                       transform=transform,
                                       target_transform=target_transform)
            dataset = ConcatDataset([voc_dataset, sbd_dataset, coco_dataset])
            dataset.num_classes = 21
            dataset.ignore_index = 255
            dataset.label_offset = 0
            return dataset

        # dataset attributes
        voc_dataset.num_classes = 21
        voc_dataset.ignore_index = 255
        voc_dataset.label_offset = 0
        return voc_dataset

    elif dataset == 'nyu':
        # NYUv2-40 dataset
        dataset = NYU(root_dir='data/nyu',
                      image_set=image_set,
                      transform=transform,
                      target_transform=target_transform)
        dataset.num_classes = 40
        dataset.ignore_index = -1
        dataset.label_offset = 1
        return dataset
