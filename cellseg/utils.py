import matplotlib.pyplot as plt
from math import ceil
from .data import DataProcessor



def show_images(dataset_object, number=None, fig_size=(20, 20), target="image"):
    """
    :param dataset_object: An object of class DataProcessor.
    :param number: Number of images to plot
    :param target: Type of images to show. One of "image" or "mask", defaults to image
    :param fig_size: Figure size, defaults to (20, 20)
    :return: A plot showing images or labels
    """
    if target not in ["image", "mask"]:
        raise ValueError(f"Target should be one of image or mask not {target}")

    if not isinstance(dataset_object, DataProcessor):
        raise TypeError(f"Expected an object of class DataProcessor not {type(dataset_object).__name__}")



    if isinstance(number, int) and number <= dataset_object.__len__:
        plt_figure = plt.figure(figsize=fig_size)
        num_cols = number / 2 if number % 2 == 0 else number / 3
        for img_index in range(number):
            subplt = plt_figure.add_subplot( 2, int(ceil(num_cols)),img_index + 1)
            # Current shape is channels first, drop first dimension (channel) for viewing
            subplt.imshow(dataset_object[img_index][target][0, :, :], cmap="gray")

    else:
        raise ValueError(f"number should be an int and less than or equal to {dataset_object.__len__} not {number}")










