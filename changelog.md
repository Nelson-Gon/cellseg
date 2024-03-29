# cellseg: Multiclass Cell Segmentation


**Version 0.1.0**

* Initialized script mode. 

* Optimized imports 

* Updated docs to reflect proper DataProcessor usage. 

* Initial tests. 

* `DataProcessor` now errors if image and mask/label lengths differ. 

* Dropped thresholding methods. Please use [pyautocv](https://github.com/Nelson-Gon/pyautocv) or any other image 
processing package of your choice. 

* Refactored `show_images` to handle follow current `DataProcessor` logic. Fixed a bug that caused rows and column
switch. 

* `DataProcessor` now returns a dictionary containing an image, its label, and index.  

* Reading stacked tiff images is no longer supported for now.

* `dir_type` was dropped in `DataProcessor`. Only provide a directory to `image_dir` containing images and masks. 

* Added data from [cytounet](https://github.com/Nelson-Gon/cytounet)

* Versioning is now automated. 

* Updated docs to show that this is a work in progress.

* Updated docs to use explicit imports. 

**version 0.0.0**

* Preserve name on PyPI

* Fixed issues with `show_images` showing blank images for masks (labels). 

* Fixed issues with `uint16` not working with `PIL`.

* `DataProcessor` can now transform images to a given target size. 

* Renamed `DataLoader` class to `DataProcessor` to avoid conflicts with `torch.utils.data.DataLoader`

* Added initial simple CNN model with a single layer

* Added `show_images` in `utils.py` to allow quick visualization of a given number of images from a given stack of
images. 

* Implemented data loaders. 

* Conceptualized project 