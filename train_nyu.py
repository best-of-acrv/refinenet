import os
import torch
from models.refinenet import refinenet50, refinenet101, refinenet152
from data_utils.transforms import get_transforms
from data_utils.datasets import get_dataset
from models.optim import get_optimiser
from helpers.arguments import get_argument_parser
from helpers.trainer import Trainer

# get general arguments
parser = get_argument_parser()
# add dataset specific arguments
parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
parser.add_argument('--load_directory', type=str, default='runs/refinenet', help='load model directory')
args = parser.parse_args()
args.save_directory = os.path.join(args.save_directory, args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)

if __name__ == '__main__':

    # Prepare training dataset
    transform, target_transform = get_transforms(crop_size=args.crop_size,
                                                 lower_scale=args.lower_scale,
                                                 upper_scale=args.upper_scale)
    dataset = get_dataset(dataset='nyu',
                          image_set='train',
                          transform=transform,
                          target_transform=target_transform)


    # Prepare validation dataset
    transform, target_transform = get_transforms(mode='eval')
    val_dataset = get_dataset(dataset='nyu',
                              image_set='test',
                              transform=transform,
                              target_transform=target_transform)

    # Initialise model and optimiser
    if args.num_resnet_layers == 50:
        model = refinenet50(num_classes=dataset.num_classes)
    elif args.num_resnet_layers == 101:
        model = refinenet101(num_classes=dataset.num_classes)
    elif args.num_resnet_layers == 152:
        model = refinenet152(num_classes=dataset.num_classes)
    else:
        print('Invalid number of ResNet layers chosen. Please choose from 50, 101 or 152 layers')

    # Get optimiser & attach to model
    model = get_optimiser(args, model)

    # try to load model (if any)
    if args.load_directory:
        model.load(log_directory=args.load_directory)

    # move model to device (if available)
    if model.cuda_available:
        model.cuda()

    # Initialise model trainer and train
    trainer = Trainer(args)
    trainer.train(args, model, dataset, val_dataset)

