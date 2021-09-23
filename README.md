# cellseg: Multiclass Cell Segmentation 

[![PyPI version](https://badge.fury.io/py/cellseg.svg)](https://badge.fury.io/py/cellseg) 
![Stage](https://www.repostatus.org/badges/latest/wip.svg)
[![Codecov](https://codecov.io/gh/Nelson-Gon/cellseg/branch/main/graph/badge.svg)](https://codecov.io/gh/Nelson-Gon/cellseg?branch=main)
[![Test Install](https://github.com/Nelson-Gon/cellseg/actions/workflows/build-package.yaml/badge.svg)](https://github.com/Nelson-Gon/cellseg/actions/workflows/build-package.yaml)
[![PyPI license](https://img.shields.io/pypi/l/cellseg.svg)](https://pypi.python.org/pypi/cellseg/) 
[![Documentation Status](https://readthedocs.org/projects/cellseg/badge/?version=latest)](https://cellseg.readthedocs.io/en/latest/?badge=latest)
[![Total Downloads](https://pepy.tech/badge/cellseg)](https://pepy.tech/project/cellseg)
[![Monthly Downloads](https://pepy.tech/badge/cellseg/month)](https://pepy.tech/project/cellseg)
[![Weekly Downloads](https://pepy.tech/badge/cellseg/week)](https://pepy.tech/project/cellseg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nelson-Gon/cellseg/graphs/commit-activity)
[![GitHub last commit](https://img.shields.io/github/last-commit/Nelson-Gon/cellseg.svg)](https://github.com/Nelson-Gon/cellseg/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/Nelson-Gon/cellseg.svg)](https://GitHub.com/Nelson-Gon/cellseg/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Nelson-Gon/cellseg.svg)](https://GitHub.com/Nelson-Gon/cellseg/issues?q=is%3Aissue+is%3Aclosed)


# Development stage

- [x] Read Tiff Images

- [x] Read Non Tiff Images

- [x] Write Data Transformers and Loaders

- [ ] Write functional model plus scripts  

- [ ] Modify model weights/layers

- [ ] Read stacked tiff images/videos 

# Introduction

`cellseg` is a PyTorch (`torch`) based deep learning package aimed at multiclass cell segmentation. 

# Installation

```shell
pip install cellseg 
```
Or if you want to build from source 

```shell
git clone git@github.com:Nelson-Gon/cellseg.git
cd cellseg
python setup.py install 

```






# Usage

## Script mode 

**View images**

```shell
python -m cellseg -d data/train -t "image" -n 4 -s 512
```

To get help 

```shell
python -m cellseg --help
#usage: __main__.py [-h] -d IMAGE_DIRECTORY -s IMAGE_SIZE -t TARGET -n NUMBER
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -d IMAGE_DIRECTORY, --image-directory IMAGE_DIRECTORY
#                        Path to image directory containing images and
#                        masks/labels
#  -s IMAGE_SIZE, --image-size IMAGE_SIZE
#                        Size of images
#  -t TARGET, --target TARGET
#                        Target images to show
#  -n NUMBER, --number NUMBER
#                        Number of images to show
```

## Programming mode 

**Importing relevant modules** 

```shell

from cellseg.data import DataProcessor
from cellseg.model import CellNet
from cellseg.utils import DataProcessor, show_images
```

**Creating a a model object**

```shell
my_model = CellNet()
```

**Load training data**

```shell
train_data = DataProcessor(image_dir="data/train/images", label_dir="data/train/images", image_suffix="tif")
```

**View loaded images or masks**

```shell
show_images(train_data, number = 8, target="image")
```

**Training**


