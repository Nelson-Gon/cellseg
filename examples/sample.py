from cellseg.data import *
from cellseg.utils import show_images

if __name__ == "__main__":
    test = DataProcessor(train_image_dir="path", train_mask_dir="path", target_size=(512, 512))

    show_images(test, number=4, target="mask")




