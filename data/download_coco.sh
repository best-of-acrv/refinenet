#!/bin/bash

# Download COCO dataset
mkdir -p coco && cd coco
curl -O http://images.cocodataset.org/zips/train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# unzip contents
echo 'Unzipping contents...'
unzip train2017.zip -d .
unzip val2017.zip -d .
unzip annotations_trainval2017.zip -d .

# remove zip files
rm train2017.zip
rm val2017.zip
rm annotations_trainval2017.zip
