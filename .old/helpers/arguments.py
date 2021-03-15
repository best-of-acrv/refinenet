import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser(description='PyTorch RefineNet')

    # General parameters
    parser.add_argument('--batch_size', type=int, default=4, help='batch size to train the segmenter model.')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for PyTorch dataloader.')

    # Model parameters
    parser.add_argument('--num_resnet_layers', type=int, default=50, help='number of resnet layers in model [50, 101, 152]')

    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='base learning rate')
    parser.add_argument('--optimiser_type', type=str, default='sgd', help='optimiser type to use for training')
    parser.add_argument('--display_interval', type=int, default=10, help='interval of displaying log to console (in iterations)')
    parser.add_argument('--eval_interval', type=int, default=1, help='interval between evaluating on validation (in epochs)')
    parser.add_argument('--snapshot_interval', type=int, default=5, help='interval between saving model snapshots (in epochs)')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use')

    return parser
