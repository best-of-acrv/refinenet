import acrv_datasets
import os
import PIL.Image as Image
import re
import torch
from torchvision import transforms

# from .datasets.coco import COCODataset
from .datasets.nyu import NYU
from .datasets.sbd import SBD
from .datasets.voc import VOC
from .models import refinenet, refinenet_lw
from .trainer import Trainer

# Param list from old mode:
# SHARED MODE:
# parser.add_argument('--batch_size', type=int, default=4, help='batch size to train the segmenter model.')
# parser.add_argument('--num_workers', type=int, default=4, help='number of workers for PyTorch dataloader.')

# # Model parameters
# parser.add_argument('--num_resnet_layers', type=int, default=50, help='number of resnet layers in model [50, 101, 152]')

# # Training parameters
# parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
# parser.add_argument('--learning_rate', type=float, default=5e-4, help='base learning rate')
# parser.add_argument('--optimiser_type', type=str, default='sgd', help='optimiser type to use for training')
# parser.add_argument('--display_interval', type=int, default=10, help='interval of displaying log to console (in iterations)')
# parser.add_argument('--eval_interval', type=int, default=1, help='interval between evaluating on validation (in epochs)')
# parser.add_argument('--snapshot_interval', type=int, default=5, help='interval between saving model snapshots (in epochs)')
# parser.add_argument('--seed', type=int, default=42, help='random seed to use')
#
# TRAIN SCRIPT
# parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
# parser.add_argument('--dataset', type=str, default='voc', help='name of dataset: choose from [nyu, voc]')
# parser.add_argument('--dataroot', type=str, default='../acrv-datasets/datasets/', help='root directory of data')
# parser.add_argument('--model_type', type=str, default='refinenet', help='model type: choose from [refinenet, refinenetlw]')
# parser.add_argument('--freeze_bn', type=bool, default=False, help='freeze bn params during training')
# parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
# parser.add_argument('--load_directory', type=str, default=None, help='load model directory')
# parser.add_argument('--snapshot_num', type=int, default=None, help='snapshot number of model (if any)')
#
# EVAL SCRIPT
# parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
# parser.add_argument('--dataset', type=str, default='voc', help='name of dataset: choose from [nyu, voc]')
# parser.add_argument('--dataroot', type=str, default='../acrv-datasets/datasets/', help='root directory of data')
# parser.add_argument('--model_type', type=str, default='refinenet', help='type of model to use. Choose from: [refinenet, refinenetlw]')
# parser.add_argument('--multi_scale_eval', type=bool, default=False, help='use multi-scale evaluation')
# parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
# parser.add_argument('--load_directory', type=str, default=None, help='load directory of model (if any)')
# parser.add_argument('--snapshot_num', type=int, default=None, help='snapshot number of model in epochs (if any)')
#
# DEPLOY SCRIPT
# parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
# parser.add_argument('--dataset', type=str, default='nyu', help='name of dataset: choose from [nyu, voc]')
# parser.add_argument('--dataroot', type=str, default='../acrv-datasets/datasets/', help='root directory of data')
# parser.add_argument('--model_type', type=str, default='refinenet', help='type of model to use. Choose from: [refinenet, refinenetlw]')
# parser.add_argument('--img_path', type=str, required=True, help='path to single image to evaluate model on')
# parser.add_argument('--multi_scale_eval', type=bool, default=False, help='use multi-scale evaluation')
# parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
# parser.add_argument('--load_directory', type=str, default=None, help='load directory of model (if any)')
# parser.add_argument('--snapshot_num', type=int, default=None, help='snapshot number of model in epochs (if any)')


class RefineNet(object):
    DATASETS = ['nyu', 'voc']
    MODEL_TYPES = ['full', 'lightweight']
    NUM_LAYERS = [50, 101, 152]
    OPTIMISER_TYPES = ['adam', 'sgd']
    WEIGHTS = ['nyu', 'voc', 'citiscapes']

    def __init__(self,
                 gpu_id=0,
                 model_type=MODEL_TYPES[0],
                 model_seed=0,
                 num_resnet_layers=NUM_LAYERS[0],
                 weights=None,
                 weights_file=None):
        # Apply sanitised arguments
        self.gpu_id = gpu_id
        self.model_type = _sanitise_arg(model_type, 'model_type',
                                        RefineNet.MODEL_TYPES)
        self.model_seed = model_seed
        self.num_resnet_layers = _sanitise_arg(num_resnet_layers,
                                               'num_resnet_layers',
                                               RefineNet.NUM_LAYERS)
        self.weights = weights
        self.weights_file = weights_file

        # Initialise the network
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.manual_seed(self.model_seed)

        if not torch.cuda.is_available():
            raise RuntimeWarning("PyTorch could not find CUDA, using CPU ...")

    def eval(self, dataset=None, output_file=None):
        pass

    def predict(self, image=None, image_file=None, output_file=None):
        pass

    def train(self,
              dataset_name,
              *,
              batch_size=4,
              dataset_dir=None,
              display_interval=10,
              eval_interval=1,
              freeze_batch_normal=False,
              learning_rate=5e-4,
              num_workers=4,
              optimiser_type=OPTIMISER_TYPES[1],
              output_directory=None,
              snapshot_interval=5):
        # Perform argument validation / set defaults
        dataset_name = _sanitise_arg(dataset_name, 'dataset',
                                     RefineNet.DATASETS)
        optimiser_type = _sanitise_arg(optimiser_type, 'optimiser_type',
                                       RefineNet.OPTIMISER_TYPES)

        # Obtain access to the dataset
        print("\nGETTING DATASET:")
        if dataset_dir is None:
            # TODO translate voc into all the required datasets
            dataset_dir = acrv_datasets.get_datasets(dataset_name)
        print("Using 'dataset_dir': %s" % dataset_dir)
        dataset = _load_dataset(dataset_name, dataset_dir, self.model_type)

        # Load in a starting model, and moving it to the device if required
        print("\nGETTING REQUESTED MODELS")
        model = _get_optimiser(
            _get_model(dataset_name, self.model_type, self.num_resnet_layers),
            self.model_type, optimiser_type, learning_rate)
        # TODO loading of previously saved model
        if model.cuda_available:
            model.cuda()

        # Start a model trainer
        print("\nPERFORMING TRAINING")
        Trainer(output_directory).train(
            model,
            dataset,
            eval_interval=eval_interval,
            snapshot_interval=snapshot_interval,
            display_interval=display_interval,
            batch_size=batch_size,
            num_workers=num_workers,
            freeze_batch_normal=freeze_batch_normal)


def _get_model(dataset_name, model_type, num_resnet_layers, pretrained='imagenet'):
    num_classes = {'nyu': 40, 'voc': 21, 'citiscapes': 19}[dataset_name]
    return {
        RefineNet.MODEL_TYPES[0]: {
            50: refinenet.refinenet50,
            101: refinenet.refinenet101,
            152: refinenet.refinenet152
        },
        RefineNet.MODEL_TYPES[1]: {
            50: refinenet_lw.refinenet_lw50,
            101: refinenet_lw.refinenet_lw101,
            152: refinenet_lw.refinenet_lw152
        }
    }[model_type][num_resnet_layers](num_classes=num_classes,
                                     pretrained=pretrained)


def _get_optimiser(model, model_type, optimiser_type, learning_rate):
    # Get encoder and decoder parameters from the model
    enc_params = []
    dec_params = []
    for k, v in model.named_parameters():
        if bool(re.match(".*conv1.*|.*bn1.*|.*layer.*", k)):
            enc_params.append(v)
        else:
            dec_params.append(v)

    # Attach the optimisers based on model_type & optimiser_type
    opt_fn = (torch.optim.SGD if optimiser_type == RefineNet.OPTIMISER_TYPES[1]
              else torch.optim.Adam)
    opt_params = ({
        'momentum':
            0.9,
        'weight_decay':
            5e-4 if model_type == RefineNet.MODEL_TYPES[0] else 1e-5
    } if optimiser_type == RefineNet.OPTIMISER_TYPES[1] else {})
    model.enc_optimiser = opt_fn(lr=learning_rate,
                                 params=enc_params,
                                 **opt_params)
    model.dec_optimiser = opt_fn(
        lr=(learning_rate if model_type == RefineNet.MODEL_TYPES[0] else 10 *
            learning_rate),
        params=dec_params,
        **opt_params)
    return model


def _get_transforms(crop_size=224, lower_scale=1.0, upper_scale=1.0):
    # Returns 4 transforms: transform train, target transform train, transform
    # eval & target transform eval
    return (transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size,
                                     scale=(lower_scale, upper_scale),
                                     interpolation=Image.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
            transforms.Compose([
                transforms.RandomResizedCrop(size=crop_size,
                                             scale=(lower_scale, upper_scale),
                                             interpolation=Image.NEAREST),
                transforms.RandomHorizontalFlip()
            ]),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]), None)


def _load_dataset(dataset_name, dataset_dir, model_type):
    # Get transformations
    (transform_train, target_transform_train, transform_eval,
     target_transform_eval) = _get_transforms(**({
         'crop_size': 400,
         'lower_scale': 0.7,
         'upper_scale': 1.3
     } if model_type == RefineNet.MODEL_TYPES[0] else {
         'crop_size': 500,
         'lower_scale': 0.5,
         'upper_scale': 2.0
     }))

    # Construct & return the dataset dictionary
    # TODO dataset dirs for VOC aren't currently handled correctly!!!
    train_args = {
            'root_dir': dataset_dir, 'image_set': 'train', 'transform': transform_train, 'target_transform': target_transform_train
            }
    eval_args = {
            'root_dir': dataset_dir, 'image_set': 'test', 'transform': transform_eval, 'target_transform': target_transform_eval
            }
    return ({
        'train': [NYU(**train_args)] * (2 if model_type == RefineNet.MODEL_TYPES[0] else 3),
        'val': NYU(**eval_args),
        'stage_epochs': ([250, 250] if model_type == RefineNet.MODEL_TYPES[0]
                         else [100, 100, 100]),
        'stage_gammas': ([0.1, 0.1] if model_type == RefineNet.MODEL_TYPES[0]
                         else [0.5, 0.5, 0.5])
    } if dataset_name == RefineNet.DATASETS[0] else {
        'train': [COCODataset(**train_args), SBD(**train_args), VOC(**train_args)],
        'val': [VOC(**eval_args)],
        'stage_epochs': [20, 50, 200],
        'stage_gammas': ([0.1 if model_type == RefineNet.MODEL_TYPES[0] else 0.5] * 3)
    })


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
