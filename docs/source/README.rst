
cellseg: Multiclass Cell Segmentation
=====================================


.. image:: https://badge.fury.io/py/cellseg.svg
   :target: https://badge.fury.io/py/cellseg
   :alt: PyPI version
 

.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :target: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Stage


.. image:: https://codecov.io/gh/Nelson-Gon/cellseg/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/Nelson-Gon/cellseg?branch=main
   :alt: Codecov


.. image:: https://github.com/Nelson-Gon/cellseg/workflows/Test-Package/badge.svg
   :target: https://github.com/Nelson-Gon/cellseg/workflows/Test-Package/badge.svg
   :alt: Test-Package


.. image:: https://travis-ci.com/Nelson-Gon/cellseg.svg?branch=main
   :target: https://travis-ci.com/Nelson-Gon/cellseg.svg?branch=main
   :alt: Travis Build


.. image:: https://img.shields.io/pypi/l/cellseg.svg
   :target: https://pypi.python.org/pypi/cellseg/
   :alt: PyPI license
 

.. image:: https://readthedocs.org/projects/cellseg/badge/?version=latest
   :target: https://cellseg.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://pepy.tech/badge/cellseg
   :target: https://pepy.tech/project/cellseg
   :alt: Total Downloads


.. image:: https://pepy.tech/badge/cellseg/month
   :target: https://pepy.tech/project/cellseg
   :alt: Monthly Downloads


.. image:: https://pepy.tech/badge/cellseg/week
   :target: https://pepy.tech/project/cellseg
   :alt: Weekly Downloads


.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Nelson-Gon/cellseg/graphs/commit-activity
   :alt: Maintenance


.. image:: https://img.shields.io/github/last-commit/Nelson-Gon/cellseg.svg
   :target: https://github.com/Nelson-Gon/cellseg/commits/main
   :alt: GitHub last commit


.. image:: https://img.shields.io/github/issues/Nelson-Gon/cellseg.svg
   :target: https://GitHub.com/Nelson-Gon/cellseg/issues/
   :alt: GitHub issues


.. image:: https://img.shields.io/github/issues-closed/Nelson-Gon/cellseg.svg
   :target: https://GitHub.com/Nelson-Gon/cellseg/issues?q=is%3Aissue+is%3Aclosed
   :alt: GitHub issues-closed


**Introduction**

``cellseg`` is a PyTorch (\ ``torch``\ ) based deep learning package aimed at multiclass cell segmentation. 

**Installation**

.. code-block:: shell

   pip install cellseg

Or if you want to build from source 

.. code-block:: shell

   git clone git@github.com:Nelson-Gon/cellseg.git
   cd cellseg
   python setup.py install

**Development stage**


* 
  [x] Read Tiff Images

* 
  [x] Read Non Tiff Images

* 
  [x] Write Data Transformers and Loaders

* 
  [ ] Write functional model 

* 
  [ ] Modify model weights/layers

* 
  [ ] Read stacked tiff images/videos 

**Usage**

.. code-block:: python


   from cellseg.data import DataProcessor
   from cellseg.model import CellNet
   from cellseg.utils import *

To create a model object:

.. code-block:: python

   my_model = CellNet()

To load data for training:

.. code-block:: python

   train_data = DataProcessor(image_dir="data/train/images", label_dir="data/train/images", image_suffix="tif")

To show images or masks that the ``DataProcessor`` class found:

.. code-block:: python

   show_images(test, 8, target="image")
   # Set target to mask to show labels/masks instead

**Training**
