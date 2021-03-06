# cellseg: Multiclass Cell Segmentation 

[![PyPI version](https://badge.fury.io/py/cellseg.svg)](https://badge.fury.io/py/cellseg) 
![Stage](https://www.repostatus.org/badges/latest/wip.svg)
[![Codecov](https://codecov.io/gh/Nelson-Gon/cellseg/branch/main/graph/badge.svg)](https://codecov.io/gh/Nelson-Gon/cellseg?branch=main)
[![Test Install](https://github.com/Nelson-Gon/cellseg/actions/workflows/build-package.yaml/badge.svg)](https://github.com/Nelson-Gon/cellseg/actions/workflows/build-package.yaml)
![Travis Build](https://travis-ci.com/Nelson-Gon/cellseg.svg?branch=main)
[![PyPI license](https://img.shields.io/pypi/l/cellseg.svg)](https://pypi.python.org/pypi/cellseg/) 
[![Documentation Status](https://readthedocs.org/projects/cellseg/badge/?version=latest)](https://cellseg.readthedocs.io/en/latest/?badge=latest)
[![Total Downloads](https://pepy.tech/badge/cellseg)](https://pepy.tech/project/cellseg)
[![Monthly Downloads](https://pepy.tech/badge/cellseg/month)](https://pepy.tech/project/cellseg)
[![Weekly Downloads](https://pepy.tech/badge/cellseg/week)](https://pepy.tech/project/cellseg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nelson-Gon/cellseg/graphs/commit-activity)
[![GitHub last commit](https://img.shields.io/github/last-commit/Nelson-Gon/cellseg.svg)](https://github.com/Nelson-Gon/cellseg/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/Nelson-Gon/cellseg.svg)](https://GitHub.com/Nelson-Gon/cellseg/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Nelson-Gon/cellseg.svg)](https://GitHub.com/Nelson-Gon/cellseg/issues?q=is%3Aissue+is%3Aclosed)

**Introduction**



`cellseg` is a PyTorch (`torch`) based deep learning package aimed at multiclass cell segmentation. 

**Installation**

```shell
pip install cellseg 
```
Or if you want to build from source 

```shell
git clone git@github.com:Nelson-Gon/cellseg.git
cd cellseg
python setup.py install 

```

**Development stage**

- [x] Read Tiff Images

- [x] Read Non Tiff Images

- [x] Write Data Transformers and Loaders

- [ ] Write functional model 

- [ ] Modify model weights/layers

- [ ] Read stacked tiff images/videos 




**Usage**

```python

from cellseg.data import DataProcessor
from cellseg.model import CellNet
from cellseg.utils import * 
```

To create a model object:

```python
my_model = CellNet()
```

To load data for training:

```python
train_data = DataProcessor(image_dir="data/train/images", label_dir="data/train/images", image_suffix="tif")
```

To show images or masks that the `DataProcessor` class found:

```python
show_images(test, 8, target="image")
# Set target to mask to show labels/masks instead 
```

**Training**


