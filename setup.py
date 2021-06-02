from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='refinenet',
      version='0.9.5',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      url='https://github.com/best-of-acrv/refinenet',
      description='RefineNet semantic image segmentation',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=[
          'acrv_datasets', 'numpy', 'pillow', 'pycocotools', 'scipy', 'six',
          'torch', 'torchvision'
      ],
      entry_points={'console_scripts': ['refinenet=refinenet.__main__:main']},
      classifiers=(
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
