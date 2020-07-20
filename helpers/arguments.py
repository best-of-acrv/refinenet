import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser(description='PyTorch RefineNet')

    # General parameters
    parser.add_argument('--batch_size', type=int, default=4, help='batch size to train the segmenter model.')
    parser.add_argument('--max_epochs', type=int, default=600, help='maximum number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for PyTorch dataloader.')

    # Dataset parameters
    parser.add_argument('--augment_data', type=bool, default=True, help='augment dataset with additional dataset')
    parser.add_argument('--crop_size', type=int, default=400, help='crop size for training,')
    parser.add_argument('--lower_scale', type=float, default=0.7, help='lower bound for random scale')
    parser.add_argument('--upper_scale', type=float, default=1.3, help='upper bound for random scale')

    # Model parameters
    parser.add_argument('--num_resnet_layers', type=int, default=50, help='number of resnet layers in model [50, 101, 152]')

    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='base learning rate')
    parser.add_argument('--optimiser_type', type=str, default='sgd', help='optimiser type to use for training')
    parser.add_argument('--display_interval', type=int, default=10, help='interval of displaying log to console')
    parser.add_argument('--eval_interval', type=int, default=5000, help='interval between evaluating on validation')
    parser.add_argument('--snapshot_interval', type=int, default=5000, help='interval between saving model snapshots')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')

    return parser
