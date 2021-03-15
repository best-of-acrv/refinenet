import numpy as np
import torch
import torch.nn.functional as F
from utils.cmap import ColourMap
from models.refinenet import refinenet101

class Segmenter:
    def __init__(self):
        # load refinenet101 model
        self.model = refinenet101(num_classes=21, pretrained='voc')
        self.cmap = ColourMap(dataset='voc')

        # move to device if cuda is available
        if self.model.cuda_available:
            self.model.cuda()

    def segment(self, im):
        # set model to eval mode for inference
        self.model.eval()
        # generate segmentation from input image
        with torch.no_grad():

            # normalise image
            im = im.astype(np.float32)
            im /= 255.0
            im -= [0.485, 0.456, 0.406]
            im /= [0.229, 0.224, 0.225]
            im = np.transpose(im, [2,0,1])
            im = torch.from_numpy(im)
            im = im.unsqueeze(dim=0)

            # move data to device if available
            if self.model.cuda_available:
                im = im.cuda()

            # forward pass through
            logits = self.model(im)

            # interpolate logits back to original image size
            prediction = F.softmax(logits, dim=1)
            prediction = F.interpolate(prediction, (im.shape[-2], im.shape[-1]), mode='bilinear')
            prediction = torch.argmax(prediction, dim=1)
            prediction = torch.squeeze(prediction)

            # convert PyTorch tensor to ndarray
            prediction = prediction.cpu().detach().numpy().astype(np.uint8)

        return prediction

    def colourise(self, pred):
        # apply colour map
        output = self.cmap.colourise(pred)

        return output