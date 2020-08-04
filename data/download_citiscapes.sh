#!/bin/bash

# Download Citiscapes dataset
mkdir -p citiscapes && cd citiscapes
wget -O citiscapes.zip https://cloudstor.aarnet.edu.au/plus/s/umyAPaYRM7etDZu/download

# Unzip contents
echo 'Unzipping contents...'
unzip citiscapes.zip -d .

# remove zip file
rm citiscapes.zip
