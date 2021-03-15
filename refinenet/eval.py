import os
import torch
from data_utils.get_dataset import get_dataset
from models.get_model import get_model
from helpers.arguments import get_argument_parser
from helpers.evaluator import Evaluator

# get general arguments
parser = get_argument_parser()
# add dataset specific arguments
parser.add_argument('--name', type=str, default='refinenet', help='custom prefix for naming model')
parser.add_argument('--dataset', type=str, default='voc', help='name of dataset: choose from [nyu, voc]')
parser.add_argument('--dataroot', type=str, default='../acrv-datasets/datasets/', help='root directory of data')
parser.add_argument('--model_type', type=str, default='refinenet', help='type of model to use. Choose from: [refinenet, refinenetlw]')
parser.add_argument('--multi_scale_eval', type=bool, default=False, help='use multi-scale evaluation')
parser.add_argument('--save_directory', type=str, default='runs', help='save model directory')
parser.add_argument('--load_directory', type=str, default=None, help='load directory of model (if any)')
parser.add_argument('--snapshot_num', type=int, default=None, help='snapshot number of model in epochs (if any)')
args = parser.parse_args()
args.save_directory = os.path.join(args.save_directory, args.name)

# GPU settings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)

if __name__ == '__main__':

    # Get dataset (train and validation) with epoch stages
    dataset = get_dataset(dataset=args.dataset, data_root=args.dataroot, model_type=args.model_type)

    # Get model
    model = get_model(args, pretrained=args.dataset)

    # load model from directory (if specified)
    if args.load_directory:
        model.load(log_directory=args.load_directory, snapshot_num=args.snapshot_num, with_optim=False)
    if model.cuda_available:
        model.cuda()

    # save samples and compute mean IU
    evaluator = Evaluator(args, multi_scale=args.multi_scale_eval)
    evaluator.sample(model, dataset['val'])
    evaluator.compute_miu(model, dataset['val'])
