from cellseg.data import *
from cellseg.utils import show_images

if __name__ == "__main__":
    test = DataProcessor(train_image_dir="D:\\train_images", train_mask_dir="D:\\train_images", target_size=(256, 256))

    show_images(test, number=4, target="mask")




