# Frequently Asked Questions

1.  Boolean value of Tensor with more than one value is ambiguous

This may be due to calling the wrong function. A way to reproduce it is as follows:

```python
model = CellSeg()
input_data = # your input data here
output = MyModel(input_data) # This will throw the above error
```

You should instead call:

```python
model(input_data)

```

2. Unable to multiply incompatible shapes 

Check the calculation of the input and output, especially after max pooling. For an input of 256 * 256, 
same padding and a 2 * 2 max pooling for example:

```python
conv1 = nn.Conv2d(1, 256, 3)
pool = nn.MaxPool2d(2)
```
To calculate the input to our dense layer, we need to consider that 

* The max pooling operation will "halve" our input in both dimensions and therefore give us an input of
`[256, 254/2,254/2]`. Therefore, for the dense layer, our input will be 256 * 127 * 127. 
  
3. DefaultCPUAllocator: not enough memory

You need to reduce the size of your input or use a computer with more computational power. 

4. `show_images` shows blank/distorted images

This is probably because you provided a single image and not a stacked (timelapse) image. Single images are currently
not supported. 

5. `show_images` shows distorted images. I have already used a stacked image.

This is most probably because the image is of data type `uint16` which unfortunately at the time is not properly 
handled with `PIL` yet `PIL` is required for `torch` transformations used in `data.py`. 

Alternatively, you may write your own data generator instead of using `DataLoader`. The image processing done in 
`DataLoader` is specific to the dataset used in building this model. 

See [also](https://github.com/python-pillow/Pillow/issues/1514), 
[this](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes) and 
[this](https://scikit-image.org/docs/dev/user_guide/data_types.html). 