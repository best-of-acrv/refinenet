# README #

This repository provides official models from the paper **RefineNet: Multi-Path Refinement Networks 
for High-Resolution Semantic Segmentation**, which is available [here](https://arxiv.org/abs/1611.06612)

> Lin, Guosheng, et al. 'Refinenet: Multi-path refinement networks for 
> high-resolution semantic segmentation.' Proceedings of the IEEE conference on 
> computer vision and pattern recognition. 2017.

Additionally, we also support the Lightweight-RefineNet models from the paper **Light-Weight RefineNet for Real-Time 
Semantic Segmentation**, which is available [here](https://arxiv.org/abs/1810.03272)

> Nekrasov, Vladimir, Chunhua Shen, and Ian Reid. "Light-Weight RefineNet for 
> Real-Time Semantic Segmentation." British Machine Vision Conference, 2018.

This repository is designed to provide out-of-the-box functionality for evaluation and training of
RefineNet and Light-Weight RefineNet models as specified in their respective papers, with as little overhead as possible. Models were adapted from
the official [RefineNet](https://github.com/guosheng/refinenet) and [Light-Weight RefineNet](https://github.com/DrSleep/light-weight-refinenet).

## Setup ##
To create the Conda environment to run code from this repository:

```
$ conda env create -f requirements.yml
```
This should set up the conda environment with all prerequisites for running this code. Activate this Conda
environment using the following command:
```
$ conda activate pytorch-refinenet
```


### Install COCO API ### 
Optional: clone and install the official COCO API Git Repository (if using the COCO dataset):
```
$ git clone https://github.com/cocodataset/cocoapi
$ cd cocoapi/PythonAPI
$ make
$ python setup.py install
```

## Datasets ##
For the datasets required for this project, please refer to [this](https://github.com/best-of-acrv/acrv-datasets). Use 
this to download and prepare NYU, Pascal VOC and COCO datasets. The data directories should appear in the following structure:
```
root_dir
|--- deploy.py
|--- eval.py
|--- train.py
acrv-datasets
|--- datasets
|------- coco
|------- nyu
|------- pascal_voc
|------- sbd
```

## Evaluation ##
To evaluate with one of the pretrained models, run ```eval.py```.
 
You can specifying the desired dataset (VOC, NYU or Citiscapes) and the desired model type (RefineNet or RefineNet-LW)
For example, to evaluate on the NYUv2 dataset using a RefineNet-101 model and generate sample
segmentation images, run the following command from the root directory:

```python eval.py --dataset=nyu --model_type=refinenet --num_resnet_layers=101```

Pretrained RefineNet models will be automatically downloaded and stored in the ```pretrained/models``` directory.
Alternatively, if you wish to load your own pretrained model, you can do this by specifying a load directory (e.g.):

```python eval.py --num_resnet_layers=50 --model_type=refinenetlw --load_directory=runs/mymodel```

Will load a Light-Weight RefineNet with a ResNet-50 backbone encoder, from the directory ``runs/mymodel``. We also support multi-scale evaluation as specified in the RefineNet paper. To enable multi-scale evaluation simply set
the flag to ```True``` (e.g.):

```python eval.py --dataset=nyu --num_resnet_layers=50 --model_type=refinenetlw --multi_scale_eval=True --load_directory=runs/mymodel```

## Training ##
To train your own RefineNet model, run ```train.py```. 

Use ``--model_type`` to choose between RefineNet and Light-Weight RefineNet. By default to assist with training, models will be preloaded with ImageNet weights 
for the backbone ResNet encoder. For example, to train on the NYUv2 dataset using a RefineNet-101 model, 
run the following command from the root directory:

```python train.py --dataset=nyu --num_resnet_layers=101 --learning_rate=0.0005```

## Deploying ##
For deploying a RefineNet model (sampling a single image for segmentation), run ```deploy.py```, e.g.:

```python deploy.py --dataset=nyu --num_resnet_layers=101 --img_path=path/to/image```


