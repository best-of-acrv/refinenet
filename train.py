import os
import torch
from data_utils.get_dataset import get_dataset
from models.get_model import get_model
from models.get_optim import get_optimisers
from helpers.arguments import get_argument_parser
from helpers.trainer import Trainer

# get general arguments
parser = get_argument_parser()
# add dataset specific arguments
parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
parser.add_argument('--dataset', type=str, default='voc', help='name of dataset: choose from [nyu, voc, citiscapes]')
parser.add_argument('--model_type', type=str, default='refinenet', help='model type: choose from [refinenet, refinenetlw]')
parser.add_argument('--freeze_bn', type=bool, default=False, help='freeze bn params during training')
parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
parser.add_argument('--load_directory', type=str, default=None, help='load model directory')
parser.add_argument('--snapshot_num', type=int, default=None, help='snapshot number of model (if any)')
args = parser.parse_args()
args.save_directory = os.path.join(args.save_directory, args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)

if __name__ == '__main__':

    # Get dataset (train and validation) with epoch stages
    dataset = get_dataset(dataset=args.dataset, model_type=args.model_type)

    # Get model
    model = get_model(args)

    # Get optimisers & attach to model
    model = get_optimisers(args, model)

    # try to load model (if any)
    if args.load_directory:
        model.load(log_directory=args.load_directory, snapshot_num=args.snapshot_num)

    # move model to device (if available)
    if model.cuda_available:
        model.cuda()

    # Initialise model trainer and train
    trainer = Trainer(args)
    trainer.train(args, model, dataset)

