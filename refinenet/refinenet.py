import acrv_datasets
import os
import torch

from .models import refinenet, refinenet_lw

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
    MODELS = ['full', 'lightweight']
    NUM_LAYERS = [50, 101, 152]
    WEIGHTS = ['nyu', 'voc', 'citiscapes']

    def __init__(self,
                 gpu_id=0,
                 model=MODELS[0],
                 model_seed=0,
                 num_resnet_layers=NUM_LAYERS[0],
                 weights=None,
                 weights_file=None):
        # Validate arguments
        if model.lower() not in RefineNet.MODELS:
            raise ValueError(
                "Invalid 'model' provided. Supported values are one of:"
                "\n\t%s" % RefineNet.MODELS)
        if num_resnet_layers not in RefineNet.NUM_LAYERS:
            raise ValueError(
                "Invalid 'num_resnet_layers' provided. Supported values are:"
                "\n\t%s" % RefineNet.NUM_LAYERS)

        # Apply arguments
        self.gpu_id = gpu_id
        self.model = model.lower()
        self.model_seed = model_seed
        self.num_resnet_layers = num_resnet_layers
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

    def train(self, dataset, learning_rate=None):
        # Obtain access to the dataset
        dataset = dataset.lower()
        print("\nGETTING DATASETS:")
        dataset_dir = acrv_datasets.get_datasets(dataset)

        # Load in a starting model, and moving it to the device if required
        print("\nGETTING REQUESTED MODELS")
        model = _get_model(dataset, self.model, self.num_resnet_layers)

        # Start a model trainer


def _get_model(dataset, model, num_resnet_layers, pretrained='imagenet'):
    num_classes = {'nyu': 40, 'voc': 21, 'citiscapes': 19}[dataset]
    return {
        RefineNet.MODELS[0]: {
            50: refinenet.refinenet50,
            101: refinenet.refinenet101,
            152: refinenet.refinenet152
        },
        RefineNet.MODELS[1]: {
            50: refinenet_lw.refinenet_lw50,
            101: refinenet_lw.refinenet_lw101,
            152: refinenet_lw.refinenet_lw152
        }
    }[model][num_resnet_layers](num_classes=num_classes, pretrained=pretrained)
