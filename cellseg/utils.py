import matplotlib.pyplot as plt
import numpy as np
from data import DataLoader


def show_images(dataset_object, stack_number=0, number=None, fig_size=(20, 20)):
    """

    :param dataset_object: An object of class DataLoader
    :param stack_number: Frame number for tiff images. Defaults to zero.
    :param number: Number of images to plot from the frame
    :param fig_size: Figure size, defaults to (20, 20)
    :return: A plot showing images from the stack number chosen.

    """
    if not isinstance(dataset_object, DataLoader):
        raise TypeError(f"Expected an object of class DataLoader not {type(dataset_object).__name__}")

    plt_figure = plt.figure(figsize=fig_size)
    get_images = dataset_object[stack_number]["image"]
    get_length = get_images.shape[0] if number is None else number
    num_cols = number / 2 if get_length % 2 == 0 else number / 3
    for index in range(get_length):
        subplt = plt_figure.add_subplot(np.int(np.ceil(num_cols)), 2, index + 1)
        subplt.imshow(get_images[index], cmap="gray")

