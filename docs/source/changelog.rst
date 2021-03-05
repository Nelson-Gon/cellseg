
cellseg: Multiclass Cell Segmentation
=====================================

**Version 0.1.0**


* 
  Update docs to show that this is a work in progress.

* 
  Update docs to use explicit imports. 

**version 0.0.0**


* 
  Preserve name on PyPI

* 
  Fixed issues with ``show_images`` showing blank images for masks (labels). 

* 
  Fixed issues with ``uint16`` not working with ``PIL``.

* 
  ``DataProcessor`` can now transform images to a given target size. 

* 
  Renamed ``DataLoader`` class to ``DataProcessor`` to avoid conflicts with ``torch.utils.data.DataLoader``

* 
  Added initial simple CNN model with a single layer

* 
  Added ``show_images`` in ``utils.py`` to allow quick visualization of a given number of images from a given stack of
  images. 

* 
  Implemented data loaders. 

* 
  Conceptualized project 
