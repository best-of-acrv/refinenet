import os
from data_utils.transforms import get_transforms
from data_utils.dataset.voc import VOC
from data_utils.dataset.sbd import SBD
from data_utils.dataset.coco import COCODataset
from data_utils.dataset.nyu import NYU

def get_dataset(dataset, data_root, model_type):

    if model_type.lower() == 'refinenet':
        # Transformations for training and validation datasets
        transform_train, target_transform_train = get_transforms(crop_size=400,
                                                                 lower_scale=0.7,
                                                                 upper_scale=1.3)
        # Prepare validation dataset
        transform_val, target_transform_val = get_transforms(mode='eval')

        if dataset.lower() == 'nyu':
            # Prepare training dataset
            nyu_dataset = NYU(root_dir=os.path.join(data_root, 'nyu'), image_set='train', transform=transform_train,
                              target_transform=target_transform_train)
            train_datasets = [nyu_dataset, nyu_dataset]
            stage_epochs = [250, 250]
            stage_gammas = [0.1, 0.1]

            # Prepare validation dataset
            val_dataset = NYU(root_dir=os.path.join(data_root, 'nyu'), image_set='test', transform=transform_val,
                              target_transform=target_transform_val)
        elif dataset.lower() == 'voc':
            # Prepare training dataset
            # VOC Dataset
            voc_dataset = VOC(root_dir=os.path.join(data_root, 'voc'), image_set='train',
                              transform=transform_train, target_transform=target_transform_train)
            # Semantic Boundaries Dataset (for augmentation)
            sbd_dataset = SBD(root_dir=os.path.join(data_root, 'sbd'), image_set='train',
                              transform=transform_train, target_transform=target_transform_train)
            # Microsoft COCO Dataset (for augmentation)
            coco_dataset = COCODataset(root_dir=os.path.join(data_root, 'coco'), image_set='train',
                                       transform=transform_train, target_transform=target_transform_train)
            train_datasets = [coco_dataset, sbd_dataset, voc_dataset]
            stage_epochs = [20, 50, 200]
            stage_gammas = [0.1, 0.1, 0.1]

            # Prepare validation dataset
            val_dataset = VOC(root_dir=os.path.join(data_root, 'voc'), image_set='val',
                              transform=transform_val, target_transform=target_transform_val)

    elif model_type.lower() == 'refinenetlw':
        # Transformations for training and validation datasets
        transform_train, target_transform_train = get_transforms(crop_size=500,
                                                                 lower_scale=0.5,
                                                                 upper_scale=2.0)
        # Prepare validation dataset
        transform_val, target_transform_val = get_transforms(mode='eval')

        if dataset.lower() == 'nyu':
            # Prepare training dataset
            nyu_dataset = NYU(root_dir=os.path.join(data_root, 'nyu'), image_set='train', transform=transform_train,
                              target_transform=target_transform_train)
            train_datasets = [nyu_dataset, nyu_dataset, nyu_dataset]
            stage_epochs = [100, 100, 100]
            stage_gammas = [0.5, 0.5, 0.5]

            # Prepare validation dataset
            val_dataset = NYU(root_dir=os.path.join(data_root, 'nyu'), image_set='test', transform=transform_val,
                              target_transform=target_transform_val)
        elif dataset.lower() == 'voc':
            # Prepare training dataset
            # VOC Dataset
            voc_dataset = VOC(root_dir=os.path.join(data_root, 'voc'), image_set='train',
                              transform=transform_train, target_transform=target_transform_train)
            # Semantic Boundaries Dataset (for augmentation)
            sbd_dataset = SBD(root_dir=os.path.join(data_root, 'sbd'), image_set='train',
                              transform=transform_train, target_transform=target_transform_train)
            # Microsoft COCO Dataset (for augmentation)
            coco_dataset = COCODataset(root_dir=os.path.join(data_root, 'coco'), image_set='train',
                                       transform=transform_train, target_transform=target_transform_train)
            train_datasets = [coco_dataset, sbd_dataset, voc_dataset]
            stage_epochs = [20, 50, 200]
            stage_gammas = [0.5, 0.5, 0.5]

            # Prepare validation dataset
            val_dataset = VOC(root_dir=os.path.join(data_root, 'voc'), image_set='val',
                              transform=transform_val, target_transform=target_transform_val)


    # dataset return as dictionary
    dataset = {}
    dataset['train'] = train_datasets
    dataset['val'] = val_dataset
    dataset['stage_epochs'] = stage_epochs
    dataset['stage_gammas'] = stage_gammas

    return dataset