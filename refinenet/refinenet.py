import os
import torch


class RefineNet(object):
    MODELS = ['full', 'lightweight']
    NUM_LAYERS = [50, 101, 152]
    WEIGHTS = ['nyu', 'voc', 'citiscapes']

    def __init__(self,
                 gpu_num=0,
                 model=MODELS[0],
                 model_seed=0,
                 num_resnet_layers=NUM_LAYERS[0],
                 weights=None,
                 weights_file=None):
        # Validate arguments
        if model.lower() not in RefineNet.MODELS:
            raise ValueError(
                "Invalid 'model' provided. Supported values are one of:"
                "\n\t%s" % RefineNet.MODELS)
        if num_resnet_layers not in RefineNet.NUM_LAYERS:
            raise ValueError(
                "Invalid 'num_resnet_layers' provided. Supported values are:"
                "\n\t%s" % RefineNet.NUM_LAYERS)

        # Apply arguments
        self.gpu_num = gpu_num
        self.model = model.lower()
        self.model_seed = model_seed
        self.num_resnet_layers = num_resnet_layers
        self.weights = weights
        self.weights_file = weights_file

        # Initialise the network
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_num)
        torch.manual_seed(self.model_seed)

        if not torch.cuda.is_available():
            raise RuntimeWarning("PyTorch could not find CUDA, using CPU ...")

    def eval(self, dataset=None, output_file=None):
        pass

    def predict(self, image=None, image_file=None, output_file=None):
        pass

    def train(self, dataset=None, learning_rate=None):
        pass
