from cellseg.data import *
from cellseg.utils import show_images

if __name__ == "__main__":
    test = DataProcessor(train_image_dir="path_here", train_mask_dir="path_here")

    show_images(test, number=4, target="mask")
