import acrv_datasets
import os
import numpy as np
import PIL.Image as Image
import re
import torch
from torchvision import transforms

from .datasets.coco import COCO
from .datasets.nyu import NYU
from .datasets.sbd import SBD
from .datasets.voc import VOC
from .evaluator import Evaluator
from .models import refinenet, refinenet_lw
from .predictor import Predictor
from .trainer import Trainer


class RefineNet(object):
    # TODO should citiscapes, coco, sbd be in here???
    COLORMAP_PRESETS = {'nyu': NYU, 'voc': VOC}
    # TODO citiscapes should be in these lists!!!
    DATASETS = ['nyu', 'voc']
    DATASET_NUM_CLASSES = {'nyu': NYU.NUM_CLASSES, 'voc': VOC.NUM_CLASSES}
    MODEL_TYPES = ['full', 'lightweight']
    NUM_LAYERS = [50, 101, 152]
    OPTIMISER_TYPES = ['adam', 'sgd']
    PRETRAINED = ['imagenet', 'nyu', 'voc']

    MODEL_MAP = {
        MODEL_TYPES[0]: {
            50: refinenet.refinenet50,
            101: refinenet.refinenet101,
            152: refinenet.refinenet152
        },
        MODEL_TYPES[1]: {
            50: refinenet_lw.refinenet_lw50,
            101: refinenet_lw.refinenet_lw101,
            152: refinenet_lw.refinenet_lw152
        }
    }

    def __init__(self,
                 *,
                 gpu_id=0,
                 load_pretrained='imagenet',
                 load_snapshot=None,
                 load_snapshot_optimiser=True,
                 model_seed=0,
                 model_type='full',
                 num_classes=21,
                 num_resnet_layers=50):
        # Apply sanitised arguments
        self.gpu_id = gpu_id
        self.model_type = _sanitise_arg(model_type, 'model_type',
                                        RefineNet.MODEL_TYPES)
        self.model_seed = model_seed
        self.num_classes = num_classes
        self.num_resnet_layers = _sanitise_arg(num_resnet_layers,
                                               'num_resnet_layers',
                                               RefineNet.NUM_LAYERS)
        self.load_pretrained = (
            None if load_pretrained is None else _sanitise_arg(
                load_pretrained, 'load_pretrained', RefineNet.PRETRAINED))
        self.load_snapshot = load_snapshot
        self.load_snapshot_optimiser = load_snapshot_optimiser

        # Try setting up GPU integration
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.manual_seed(self.model_seed)
        if not torch.cuda.is_available():
            raise RuntimeWarning("PyTorch could not find CUDA, using CPU ...")

        # Load the model based on the specified parameters
        self.model = None
        if self.load_snapshot:
            print("\nLOADING MODEL FROM SNAPSHOT:")
            self.model = _from_snapshot(self.load_snapshot,
                                        self.load_snapshot_optimiser)
        else:
            print("\nLOADING MODEL FROM PRE-TRAINED WEIGHTS:")
            self.model = _get_model(self.model_type,
                                    self.num_resnet_layers,
                                    self.num_classes,
                                    pretrained=self.load_pretrained)
        self.model.cuda()

    def evaluate(self,
                 dataset_name,
                 *,
                 dataset_dir=None,
                 multi_scale=False,
                 output_directory='./eval_output',
                 output_images=False):
        # Perform argument validation
        dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                     RefineNet.DATASETS)

        # Load in the dataset
        dataset = _load_dataset(dataset_name, dataset_dir, self.model_type)

        # Perform the requested evaluation
        e = Evaluator(multi_scale=multi_scale,
                      output_directory=output_directory,
                      output_images=output_images)
        e.sample(self.model, dataset['val'])
        e.compute_miu(self.model, dataset['val'])

    def predict(self,
                *,
                colour_map_preset='voc',
                image=None,
                image_file=None,
                multi_scale=False,
                output_file=None):
        # Handle input arguments
        cmap_preset = _sanitise_arg(colour_map_preset, 'colour_map_preset',
                                    RefineNet.COLORMAP_PRESETS)
        cmap_preset = RefineNet.COLORMAP_PRESETS[cmap_preset]
        if image is None and image_file is None:
            raise ValueError("Only one of 'image' or 'image_file' can be "
                             "used in a call, not both.")
        elif image is not None and image_file is not None:
            raise ValueError("Either 'image' or 'image_file' must be provided")
        if output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Construct the input image
        img = (np.array(Image.open(image_file).convert('RGB'))
               if image_file else image)
        img = _get_transforms()[2](img)

        # Perform the forward pass
        out = Predictor(multi_scale=multi_scale).predict(
            img,
            self.model,
            colour_map=cmap_preset.COLOUR_MAP,
            label_offset=cmap_preset.LABEL_OFFSET)

        # Save the file if requested, & return the output
        if output_file:
            Image.fromarray(out).save(output_file)
        return out

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
              optimiser_type='sgd',
              output_directory=os.path.expanduser('~/refinenet-output'),
              snapshot_interval=5):
        # Perform argument validation / set defaults
        dataset_name = _sanitise_arg(dataset_name, 'dataset_name',
                                     RefineNet.DATASETS)
        optimiser_type = _sanitise_arg(optimiser_type, 'optimiser_type',
                                       RefineNet.OPTIMISER_TYPES)
        dataset_num_classes = RefineNet.DATASET_NUM_CLASSES.get(
            dataset_name, 0)
        if self.model.num_classes != dataset_num_classes:
            raise ValueError(
                "Can't train using a dataset with '%d' classes, when your "
                "model has been created for '%d' classes." %
                (dataset_num_classes, self.model.num_classes))

        # Load in the dataset
        dataset = _load_dataset(dataset_name, dataset_dir, self.model_type)

        # Attach new optimiser if required (and resend to GPU???)
        if (self.model.enc_optimiser is None or
                self.model.dec_optimiser is None):
            print("\nATTACHING NEW OPTIMISERS:")
            _attach_new_optimiser(self.model, self.model_type, optimiser_type,
                                  learning_rate)
            if self.model.cuda_available:
                self.model.cuda()
            print("\tDone.")

        # Start a model trainer
        print("\nPERFORMING TRAINING:")
        Trainer(output_directory).train(
            self.model,
            dataset,
            eval_interval=eval_interval,
            snapshot_interval=snapshot_interval,
            display_interval=display_interval,
            batch_size=batch_size,
            num_workers=num_workers,
            freeze_batch_normal=freeze_batch_normal)


def _attach_new_optimiser(model, model_type, optimiser_type, learning_rate):
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
    model.optimiser_type = optimiser_type
    model.learning_rate = learning_rate


def _from_snapshot(snapshot_filename, load_optim_state=True):
    print('Loading model from:\n\t%s' % snapshot_filename)
    model_data = torch.load(snapshot_filename)

    # Create a new model with settings that match the snapshot
    md = model_data['model_metadata']
    model_type = _sanitise_arg(md['type'], 'model_metadata.type',
                               RefineNet.MODEL_TYPES)
    model = RefineNet.MODEL_MAP[model_type][_sanitise_arg(
        md['num_layers'], 'model_metadata.num_layers',
        RefineNet.NUM_LAYERS)](num_classes=md['num_classes'], pretrained=None)
    model.name = 'snapshot_%s' % os.path.basename(snapshot_filename)
    _attach_new_optimiser(
        model, model_type,
        _sanitise_arg(md['optimiser_type'], 'model_metadata.optimiser_type',
                      RefineNet.OPTIMISER_TYPES), md['learning_rate'])

    # Load requested state from the snapshot (moving to CUDA where appropriate)
    model.load_state_dict(model_data['weights'], strict=False)
    if load_optim_state:
        model.enc_optimiser.load_state_dict(model_data['enc_optimiser'])
        model.dec_optimiser.load_state_dict(model_data['dec_optimiser'])
        if model.cuda_available:
            for o in [model.enc_optimiser, model.dec_optimiser]:
                for state in o.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    # TODO should probably pass the current iteration back somehow
    # curr_iteration = model_data['global_iteration']
    return model


def _get_model(model_type,
               num_resnet_layers,
               num_classes,
               pretrained='imagenet'):
    return RefineNet.MODEL_MAP[model_type][num_resnet_layers](
        num_classes=num_classes, pretrained=pretrained)


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


def _load_dataset(dataset_name, dataset_dir, model_type, quiet=False):
    # Print some verbose information
    if not quiet:
        print("\nGETTING DATASET:")
    if dataset_dir is None:
        # TODO translate voc into all the required datasets (i.e. this
        # should handle multiple dataset_dirs)
        dataset_dir = acrv_datasets.get_datasets(dataset_name)
    if not quiet:
        print("Using 'dataset_dir': %s" % dataset_dir)

    # Get transformations
    (transform_train, target_transform_train, transform_eval,
     target_transform_eval) = _get_transforms(**({
         'crop_size': 400,
         'lower_scale': 0.7,
         'upper_scale': 1.3
     } if model_type == 'full' else {
         'crop_size': 500,
         'lower_scale': 0.5,
         'upper_scale': 2.0
     }))

    # Construct & return the dataset dictionary
    # TODO dataset dirs for VOC aren't currently handled correctly!!!
    train_args = {
        'root_dir': dataset_dir,
        'image_set': 'train',
        'transform': transform_train,
        'target_transform': target_transform_train
    }
    eval_args = {
        'root_dir': dataset_dir,
        'image_set': 'test',
        'transform': transform_eval,
        'target_transform': target_transform_eval
    }
    return ({
        'train': [NYU(**train_args)] * (2 if model_type == 'full' else 3),
        'val':
            NYU(**eval_args),
        'stage_epochs':
            ([250, 250] if model_type == 'full' else [100, 100, 100]),
        'stage_gammas':
            ([0.1, 0.1] if model_type == 'full' else [0.5, 0.5, 0.5])
    } if dataset_name == 'nyu' else {
        'train': [],
        # [COCO(**train_args),
        #  SBD(**train_args),
        #  VOC(**train_args)],
        'val': VOC(**eval_args),
        'stage_epochs': [20, 50, 200],
        'stage_gammas': ([0.1 if model_type == 'full' else 0.5] * 3)
    })


def _sanitise_arg(value, name, supported_list):
    ret = value.lower() if type(value) is str else value
    if ret not in supported_list:
        raise ValueError("Invalid '%s' provided. Supported values are one of:"
                         "\n\t%s" % (name, supported_list))
    return ret
