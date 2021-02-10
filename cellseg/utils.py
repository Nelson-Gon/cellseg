import matplotlib.pyplot as plt
import numpy as np
from data import DataProcessor


def show_images(dataset_object, stack_number=0, number=None, fig_size=(20, 20), target="image"):
    """

    :param dataset_object: An object of class DataLoader
    :param stack_number: Frame number for tiff images. Defaults to zero.
    :param number: Number of images to plot from the frame
    :param target: Type of images to show. One of "image" or "mask"
    :param fig_size: Figure size, defaults to (20, 20)
    :return: A plot showing images from the stack number chosen.

    """
    if not isinstance(dataset_object, DataProcessor):
        raise TypeError(f"Expected an object of class DataLoader not {type(dataset_object).__name__}")

    plt_figure = plt.figure(figsize=fig_size)
    get_images = dataset_object[stack_number][target]
    get_length = get_images.shape[0] if number is None else number
    num_cols = number / 2 if get_length % 2 == 0 else number / 3
    for index in range(get_length):
        subplt = plt_figure.add_subplot(np.int(np.ceil(num_cols)), 2, index + 1)
        # Current shape is channels first, drop first dimension (channel) for viewing
        subplt.imshow(get_images[index][0, :, :], cmap="gray")


