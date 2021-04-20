import argparse
import shutil
import sys
import textwrap


class ShowNewlines(argparse.RawDescriptionHelpFormatter):

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
    w = shutil.get_terminal_size().columns
    p = argparse.ArgumentParser(
        formatter_class=ShowNewlines,
        description="RefineNet semantic image segmentation.\n\n"
        "Dataset interaction is performed through the acrv_datasets package. "
        "Please see it for details on downloading datasets, accessing them, "
        "and changing where they are stored.\n\n"
        "For full documentation of RefineNet, please see "
        "https://github.com/best-of-acrv/refinenet.")
    sp = p.add_subparsers(help="RefineNet modes of operation")

    p_eval = sp.add_parser(
        'evaluate',
        help="Evaluate a model's performance against a specific dataset")
    p_eval.add_argument(
        'dataset_name',
        help="Name of the dataset to use (see "
        "https://github.com/best-of-acrv/acrv-datasets for valid names)")
    p_eval.add_argument(
        '--dataset-dir',
        default=None,
        help=
        "Custom directory to search for datasets instead of 'acrv_datasets' current default"
    )

    p_pred = sp.add_parser('predict',
                           help="Use a model to predict image segmentation "
                           "from a given input image")

    p_train = sp.add_parser('train',
                            help="Train a model from a previous starting "
                            "point using a specific dataset")

    args = p.parse_args()

    # Print help if no args provided
    if len(sys.argv) == 1:
        p.print_help()
        return

    # Run requested RefineNet operations
    print(args)


if __name__ == '__main__':
    main()
