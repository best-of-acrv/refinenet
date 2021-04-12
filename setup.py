from setuptools import Extension, find_packages, setup
from setuptools.command.install import install
import subprocess

import time

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='refinenet',
      version='0.9.0',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      description='RefineNet semantic segmentation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=[
          'acrv_datasets', 'numpy', 'pillow', 'pycocotools', 'scipy', 'six',
          'torch', 'torchvision'
      ],
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
