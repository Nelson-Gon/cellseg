import matplotlib.pyplot as plt
import numpy as np
from .data import DataProcessor
from skimage.filters import threshold_multiotsu, threshold_li


def show_images(dataset_object, number=None, fig_size=(20, 20), target="image"):
    """
    :param dataset_object: An object of class DataProcessor.
    :param number: Number of images to plot
    :param target: Type of images to show. One of "image" or "mask", defaults to image
    :param fig_size: Figure size, defaults to (20, 20)
    :return: A plot showing images or labels
    """
    if not isinstance(dataset_object, DataProcessor):
        raise TypeError(f"Expected an object of class DataLoader not {type(dataset_object).__name__}")

    plt_figure = plt.figure(figsize=fig_size)
    try:
        isinstance(number, int)
        # Number should not exceed the total number of images we have
        number <= len(dataset_object.image_list)

    except TypeError as err:
        raise

    else:
        num_cols = number / 2 if number % 2 == 0 else number / 3
        for img_index in range(number):
            subplt = plt_figure.add_subplot( 2, np.int(np.ceil(num_cols)),img_index + 1)
            # Current shape is channels first, drop first dimension (channel) for viewing
            subplt.imshow(dataset_object[img_index][target][0, :, :], cmap="gray")












