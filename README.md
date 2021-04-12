<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# RefineNet: high-res semantic image segmentation

[![Best of ACRV Repository](https://img.shields.io/badge/collection-best--of--acrv-%23a31b2a)](https://roboticvision.org/best-of-acrv)

RefineNet is a generic multi-path refinement network for high-resolution semantic image segmentation and general dense prediction tasks on images. It achieves high-resolution prediction by explicitly exploiting all the information available along the down-sampling process and using long-range residual connections.

<p align="center">
<img alt="RefineNet sample image on PASCAL VOC dataset" src="https://github.com/best-of-acrv/refinenet/raw/develop/docs/refinenet_sample.png" />
</p>

This repository contains an open-source implementation of RefineNet in Python, with both the official and lightweight network models from our publications. The package provides PyTorch implementations for training, evaluation, and deployment within systems. The package is easily installable with `conda`, and can also be installed via `pip` if you'd prefer to manually handle dependencies.

Our code is free to use, and licensed under BSD-3. We simply ask that you [cite our work](#citing-our-work) if you use RefineNet in your own research.

[![@youtube RefineNet Results on the CityScapes dataset](https://github.com/best-of-acrv/refinenet/raw/develop/docs/refinenet_video.jpg)](https://www.youtube.com/watch?v=L0V6zmGP_oQ)

## Related resources

This repository brings the work from a number of sources together. Please see the links below for further details:

- our original paper: ["RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation"](#citing-our-work)
- our paper introducing the lightweight version: ["Light-Weight RefineNet for Real-Time Semantic Segmentation"](#citing-out-work)
- the original MATLAB implementation: [https://github.com/guosheng/refinenet](https://github.com/guosheng/refinenet)
- Vladimir Nekrasov's PyTorch port of RefineNet: [https://github.com/DrSleep/refinenet-pytorch](https://github.com/DrSleep/refinenet-pytorch)
- Vladimir Nekrasov's PyTorch port of lightweight RefineNet: [https://github.com/DrSleep/light-weight-refinenet](https://github.com/DrSleep/light-weight-refinenet)

## Installing RefineNet

We offer three methods for installing RefineNet:

1. [Through our Conda package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs RefineNet and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies

### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and are inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). From there, simply run:

```
u@pc:~$ conda install refinenet
```

You can see a list of our Conda dependencies in the [`./requirements.yml`](./requirements.yml) file.

### Pip

Before installing via `pip`, you must have the following system dependencies installed:

- CUDA
- TODO the rest of this list

Then RefineNet, and all its Python dependencies can be installed via:

```
u@pc:~$ pip install refinenet
```

### From source

Installing from source is very similar to the `pip` method above due to RefineNet only containing Python code. Simply clone the repository, enter the directory, and install via `pip` in editable mode:

```
u@pc:~$ pip install -e .
```

Editable mode allows you to immediately use any changes you make to RefineNet's code in your local Python ecosystem.

## Using RefineNet

Once installed, RefineNet can be used directly from the command line using Python

TODO: add details for quickstart scripts that run directly from the command line

Once installed, RefineNet can be used like any other Python package. It consists of a `RefineNet` class with three main functions for training, evaluation, and deployment. Below are some examples to help get you started with RefineNet:

## RefineNet API

The package also includes a full Python API that allows you to use RefineNet directly in your own projects.

TODO: documentation link?

The snippet below shows a number of examples of how to use RefineNet with your own projects:

```python
from refinenet import RefineNet

# Initialise a full RefineNet network with no pre-trained model
r = RefineNet()

# Initialise a standard RefineNet network with a model pre-trained on NYU
r = RefineNet(model_type='full', load_pretrained='nyu')

# Initialise a lightweight RefineNet network with 40 classes
r = RefineNet(model='lightweight', num_classes=40)

# Load a previous snapshot from a 152 layer network
r = RefineNet(load_snapshot='/path/to/snapshot', num_resnet_layers=152)

# Train a new model on the NYU dataset with a custom learning rate
r.train('nyu', learning_rate=0.0005)

# Train a model with the adam optimiser & 8 workers, saving output to ~/output
r.train('voc', optimiser_type='adam', num_workers=8,
        output_directory='~/output')

# Get a predicted segmentation as a NumPy image, given an input NumPy image
segmentation_image = r.predict(image=my_image)

# Save a segmentation image to file, given an image from another image file
r.predict(image_file='/my/prediction.jpg',
          output_file='/my/segmentation/image.jpg')

# Evaluate your model's performance on the voc dataset, & save the results with
# images
r.eval('voc', output_directy='/my/results.json', output_images=True)
```

## Citing our work

If using RefineNet in your work, please cite [our original CVPR paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf):

```bibtex
@InProceedings{Lin_2017_CVPR,
author = {Lin, Guosheng and Milan, Anton and Shen, Chunhua and Reid, Ian},
title = {RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation},
booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

Please also cite [our BMVC paper](http://bmvc2018.org/contents/papers/0494.pdf) on Light-Weight RefineNet if using the lightweight models:

```bibtex
@article{nekrasov2018light,
  title={Light-weight refinenet for real-time semantic segmentation},
  author={Nekrasov, Vladimir and Shen, Chunhua and Reid, Ian},
  journal={arXiv preprint arXiv:1810.03272},
  year={2018}
}
```
