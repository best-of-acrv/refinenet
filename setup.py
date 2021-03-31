from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='refinenet',
      version='0.1.0',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      description='Refinenet semantic segmentation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['acrv_datasets'],
      classifiers=(
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
