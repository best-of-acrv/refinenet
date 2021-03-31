import torch
import os, sys
from six.moves import urllib

from ..helpers import cache_location


def download_model(model_name, model_url, model_dir=None, map_location=None):
    if model_dir is None:
        # Try use PyTorch's recommendations by default, otherwise fallback to
        # our package level cache
        model_dir = os.getenv('TORCH_MODEL_ZOO')
        if model_dir is None and os.getenv('TORCH_HOME') is not None:
            model_dir = os.path.join(
                os.path.expanduser(os.getenv('TORCH_HOME')), 'models')
        if model_dir is None:
            model_dir = os.path.join(cache_location(), 'pretrained')
            print("Falling back to package cache for storing "
                  "pretrained models:\n\t%s" % model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading to {}\n'.format(cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def find_snapshot(snapshot_dir, snapshot_num=None):
    found_snapshot = False
    if os.path.exists(snapshot_dir):
        print('Found snapshot directory!')
        snapshots = sorted(os.listdir(snapshot_dir))
        if snapshot_num is not None:
            for snapshot in snapshots:
                if str(snapshot_num) in snapshot:
                    model_name = snapshot
                    found_snapshot = True
                    break
        else:
            model_name = snapshots[-1]
            found_snapshot = True
        if found_snapshot:
            print('Found snapshot: Loading snapshot ' + model_name + '...')
        else:
            print('Snapshot number does not exist! Please choose from:')
            print(snapshots)
            exit()
    else:
        print('Did not find snapshot directory! Training from scratch...')
        model_name = None

    return model_name
