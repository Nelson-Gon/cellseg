# DataProcessor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from skimage.io import imread
from PIL import Image
import numpy
import torch
import glob
import random
from cv2 import convertScaleAbs


class DataProcessor(Dataset):
    def __init__(self, image_dir, dir_type="image", target_size=(512, 512), image_suffix="tif"):
        """

        :param image_dir: Path to images directory. Either images or masks dir
        :param dir_type: One of "image" or "mask" Defaults to image
        :param target_size: Target size to resize to. Defaults to (512, 512).
        This was found to be the best for data preservation.
        :param image_suffix: Image suffix of the images. Defaults to "tif"

        :return An object of class DataProcessor ---> torch.utils.data.Dataset

        """
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        self.dir_type = dir_type
        self.target_size = target_size
        self.image_list = sorted(glob.glob(self.image_dir + "/*." + self.image_suffix))

    def __len__(self):
        return len(self.image_list)

    def to_uint8(self, image):
        if self.dir_type == "mask":
            uint8_img = convertScaleAbs(image, alpha=255.0 / 65535.0)
        else:
            uint8_img = convertScaleAbs(image)

        return uint8_img




    def transform(self, image):
        """

        :param image: Stacked image eg (71, 1200, 1200)
        :param mask: Stacked mask eg (71, 1200, 1200)
        :return: Transformed images that have been randomly cropped, flipped, and resized to the target size.

        """
        resize_image = transforms.Resize(size=self.target_size)
        image = resize_image(image)
        # mask = resize_image(mask)
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7

        # Random cropping

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.target_size)
        image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            # mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            # mask = TF.vflip(mask)

        # Transform to tensor

        image = TF.to_tensor(image)
        # mask = TF.to_tensor(mask)
        return image

    def __getitem__(self, img_index):
        """

        :param img_index: Index of the image
        :return: Image
        """

        if torch.is_tensor(img_index):
            img_index = img_index.tolist()
        if self.image_suffix == "tif":
            final_images = imread(self.image_list[img_index], plugin="pil")

        else:
            final_images = imread(self.image_list[img_index])

        # # Convert images to PIL/Tensor
        # # List comprehension since we have image stacks.
        # # uint16 doesn't work with resizing --> convert to uint8 but preserve information
        # # Given the data, masks do not work well with alpha scaling


        return list(map(lambda x: self.transform(Image.fromarray(self.to_uint8(x))), final_images))



