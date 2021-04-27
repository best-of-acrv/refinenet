import argparse
import os
import sys
import textwrap

from .refinenet import RefineNet


class ShowNewlines(argparse.ArgumentDefaultsHelpFormatter,
                   argparse.RawDescriptionHelpFormatter):

    def _fill_text(self, text, width, indent):
        return ''.join([
            indent + i for ii in [
                textwrap.fill(
                    s, width, drop_whitespace=False, replace_whitespace=False)
                for s in text.splitlines(keepends=True)
            ] for i in ii
        ])


def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(
        prog='refinenet',
        formatter_class=ShowNewlines,
        description="RefineNet semantic image segmentation.\n\n"
        "Dataset interaction is performed through the acrv_datasets package. "
        "Please see it for details on downloading datasets, accessing them, "
        "and changing where they are stored.\n\n"
        "For full documentation of RefineNet, please see "
        "https://github.com/best-of-acrv/refinenet.")

    p_parent = argparse.ArgumentParser(add_help=False)
    p_parent.add_argument('--gpu-id',
                          default=0,
                          help="ID of GPU to use for model")
    p_parent.add_argument('--model-type',
                          default='full',
                          choices=RefineNet.MODEL_TYPES,
                          help="Whether to use a full or lightweight model")
    p_parent.add_argument('--model-seed',
                          default=0,
                          help="Seed used for model training")
    p_parent.add_argument(
        '--num-classes',
        default=21,
        help="Number of object classes supported by the model")
    p_parent.add_argument('--num-resnet-layers',
                          default=50,
                          choices=RefineNet.NUM_LAYERS,
                          help="Number of layers for the resnet model")
    p_parent.add_argument('--load-pretrained',
                          default='imagenet',
                          choices=RefineNet.PRETRAINED,
                          help="Load these pre-trained weights in at startup")
    p_parent.add_argument('--no-pretrained',
                          default=False,
                          action='store_true',
                          help="Start with randomly initialised weights "
                          "(overrides --load-pretrained). "
                          "Note: (training will take a LONG time)")
    p_parent.add_argument('--load-snapshot',
                          default=None,
                          help="Snapshot location from which to load weights "
                          "(overrides --load-pretrained and --no-pretrained)")
    p_parent.add_argument('--no-snapshot-optimiser',
                          default=False,
                          action='store_true',
                          help="Don't load optimiser state when loading "
                          "from a snapshot")
    sp = p.add_subparsers()

    p_eval = sp.add_parser('evaluate',
                           parents=[p_parent],
                           formatter_class=ShowNewlines,
                           help="Evaluate a model's performance against a "
                           "specific dataset")
    p_eval.add_argument('dataset_name',
                        help="Name of the dataset to use (run "
                        "'acrv_datasets --supported-datasets' to see valid "
                        "names)")
    p_eval.add_argument('--dataset-dir',
                        default=None,
                        help="Search this directory for datasets instead "
                        "of the current default in 'acrv_datasets'")
    p_eval.add_argument('--multi-scale',
                        default=False,
                        action='store_true',
                        help="Generate predictions from multiple image scales")
    p_eval.add_argument('--output-directory',
                        default='.',
                        help="Directory to save evaluation results")
    p_eval.add_argument('--output-images',
                        default=False,
                        help="Save prediction images as part of the results")

    p_pred = sp.add_parser('predict',
                           formatter_class=ShowNewlines,
                           help="Use a model to predict image segmentation "
                           "from a given input image")
    p_pred.add_argument('image_file', help="Filename for input image")
    p_pred.add_argument('--colour-map',
                        default='nyu',
                        choices=RefineNet.COLORMAP_PRESETS,
                        help="Colour map to use for prediction output")
    p_pred.add_argument('--multi-scale',
                        default=False,
                        action='store_true',
                        help="Generate predictions from multiple image scales")
    p_pred.add_argument('--output-file',
                        default='./output.jpg',
                        help="Filename used for saving the output image")

    p_train = sp.add_parser('train',
                            formatter_class=ShowNewlines,
                            help="Train a model from a previous starting "
                            "point using a specific dataset")
    p_train.add_argument('dataset_name',
                         help="Name of the dataset to use (run "
                         "'acrv_datasets --supported-datasets' to see valid "
                         "names)")
    p_train.add_argument('--batch-size',
                         default=4,
                         help="Batch size to use for training")
    p_train.add_argument('--dataset-dir',
                         default=None,
                         help="Search this directory for datasets instead "
                         "of the current default in 'acrv_datasets'")
    p_train.add_argument('--display-interval',
                         default=10,
                         help="TODOBatch size to use for training")
    p_train.add_argument('--eval-interval', default=1, help="TODO")
    p_train.add_argument('--freeze-batch-normal', default=False, help="TODO")
    p_train.add_argument('--learning-rate',
                         default=5e-4,
                         help="Learning rate for model while training")
    p_train.add_argument('--num-workers',
                         default=4,
                         help="Number of workers to use while training")
    p_train.add_argument('--optimiser-type',
                         default='sqd',
                         choices=RefineNet.OPTIMISER_TYPES,
                         help="Type of optimiser to use ")
    p_train.add_argument('--output-directory',
                         default=os.path.expanduser('~/refinenet-output'),
                         help="Location where snapshots and training progress "
                         "will be stored")
    p_train.add_argument('--snapshot-interval', default=5, help="TODO")

    args = p.parse_args()

    # Print help if no args provided
    if len(sys.argv) == 1:
        p.print_help()
        return

    # Run requested RefineNet operations
    print(args)


if __name__ == '__main__':
    main()
