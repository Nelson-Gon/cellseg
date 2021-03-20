import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#sys.path.append(0, )
if __name__ == "__main__":

    from cellseg.data import *
    from cellseg.utils import show_images

    test = DataProcessor(image_dir="data/train/images", label_dir="data/train/masks",
                         target_size=(512, 512))

    show_images(test, 6, target="image")

    show_images(test, 8, target="mask")



