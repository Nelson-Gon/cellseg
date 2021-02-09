# cellseg: Quantify and Classify Cells

**version 0.1.0**

* `DataProcessor` can now transform images to a given target size. 

* Renamed `DataLoader` class to `DataProcessor` to avoid conflicts with `torch.utils.data.DataLoader`

* Added initial simple CNN model with a single layer

* Added `show_images` in `utils.py` to allow quick visualization of a given number of images from a given stack of
images. 

* Implemented data loaders. 

* Conceptualized project 