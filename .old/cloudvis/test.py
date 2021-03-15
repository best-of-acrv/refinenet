import os
import cv2
from PIL import Image
from cloudvis.segmenter import Segmenter

if __name__ == '__main__':

    # read input image
    im = cv2.imread('data/pascal_voc/VOCdevkit/VOC2012/JPEGImages/2007_000027.jpg')

    # Get model for refinenet
    segmenter = Segmenter()

    # generate segmentation image
    pred = segmenter.segment(im)

    # and colourise
    output = segmenter.colourise(pred)

    # save prediction image
    im = Image.fromarray(output)
    im.save(os.path.join('cloudvis', 'output.png'))
