
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


.. image:: https://github.com/Nelson-Gon/cellseg/actions/workflows/build-package.yaml/badge.svg
   :target: https://github.com/Nelson-Gon/cellseg/actions/workflows/build-package.yaml
   :alt: Test Install


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


Development stage
=================


* 
  [x] Read Tiff Images

* 
  [x] Read Non Tiff Images

* 
  [x] Write Data Transformers and Loaders

* 
  [ ] Write functional model plus scripts  

* 
  [ ] Modify model weights/layers

* 
  [ ] Read stacked tiff images/videos 

Introduction
============

``cellseg`` is a PyTorch (\ ``torch``\ ) based deep learning package aimed at multiclass cell segmentation. 

Installation
============

.. code-block:: shell

   pip install cellseg

Or if you want to build from source 

.. code-block:: shell

   git clone git@github.com:Nelson-Gon/cellseg.git
   cd cellseg
   python setup.py install

Usage
=====

Script mode
-----------

**View images**

.. code-block:: shell

   python -m cellseg -d data/train -t "image" -n 4 -s 512

To get help 

.. code-block:: shell

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

Programming mode
----------------

**Importing relevant modules** 

.. code-block:: shell


   from cellseg.data import DataProcessor
   from cellseg.model import CellNet
   from cellseg.utils import DataProcessor, show_images

**Creating a a model object**

.. code-block:: shell

   my_model = CellNet()

**Load training data**

.. code-block:: shell

   train_data = DataProcessor(image_dir="data/train/images", label_dir="data/train/images", image_suffix="tif")

**View loaded images or masks**

.. code-block:: shell

   show_images(train_data, number = 8, target="image")

**Training**
