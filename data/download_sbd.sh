#!/bin/bash

# Download SBD dataset
wget -O sbd.tar http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

# Untar contents
echo 'Decompressing contents...'
mkdir -p sbd
tar -xvf sbd.tar -C sbd

# remove tar file
rm sbd.tar
