#!/bin/bash

# Download NYU dataset
wget -O nyu.zip https://cloudstor.aarnet.edu.au/plus/s/sxDddyNYmyFDEfJ/download

# Unzip contents
echo 'Unzipping contents...'
unzip nyu.zip -d .

# remove zip file
rm nyu.zip
