import os
import torch
from models.refinenet import refinenet50, refinenet101, refinenet152
from data_utils.transforms import get_transforms
from data_utils.datasets import get_dataset
from helpers.arguments import get_argument_parser
from helpers.evaluator import Evaluator

# get general arguments
parser = get_argument_parser()
# add dataset specific arguments
parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
parser.add_argument('--multi_scale_eval', type=bool, default=False, help='use multi-scale evaluation')
parser.add_argument('--load_directory', type=str, default=None, help='load directory of model (if any)')
parser.add_argument('--snapshot_num', type=int, default=40000, help='snapshot number of model (if any)')
args = parser.parse_args()
args.save_directory = os.path.join('runs', args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)

if __name__ == '__main__':

    # Prepare evaluation dataset
    transform, target_transform = get_transforms(mode='eval')
    dataset = get_dataset(dataset='voc',
                          image_set='val',
                          transform=transform,
                          target_transform=target_transform)

    # Initialise model
    if args.num_resnet_layers == 50:
        model = refinenet50(num_classes=dataset.num_classes, pretrained='voc')
    elif args.num_resnet_layers == 101:
        model = refinenet101(num_classes=dataset.num_classes, pretrained='voc')
    elif args.num_resnet_layers == 152:
        model = refinenet152(num_classes=dataset.num_classes, pretrained='voc')
    else:
        print('Invalid number of ResNet layers chosen. Please choose from 50, 101 or 152 layers')

    # load model from directory (if specified)
    if args.load_directory:
        model.load(log_directory=args.load_directory, snapshot_num=args.snapshot_num, with_optim=False)
    if model.cuda_available:
        model.cuda()

    # save samples and compute mean IU
    evaluator = Evaluator(args, multi_scale=args.multi_scale_eval)
    evaluator.sample(model, dataset)
    evaluator.compute_miu(model, dataset)