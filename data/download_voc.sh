#!/bin/bash

# Download VOC dataset
wget -O voc.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Untar contents
echo 'Decompressing contents...'
mkdir -p pascal_voc
tar -xvf voc.tar -C pascal_voc

# remove tar file
rm voc.tar
