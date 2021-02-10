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
    def __init__(self, train_image_dir, train_mask_dir, target_size=(512, 512), image_suffix="tif"):
        """

        :param train_image_dir: Path to train images directory
        :param train_mask_dir: Path to train masks (labels) directory
        :param target_size: Target size to resize to. Defaults to (512, 512). This was found to be the best for data
        preservation.
        :param image_suffix: Image suffix of the images. Defaults to "tif"

        :return An object of class DataProcessor ---> super class Dataset

        """
        self.train_image_dir = train_image_dir
        self.image_suffix = image_suffix
        self.train_mask_dir = train_mask_dir
        self.target_size = target_size
        self.train_image_list = sorted(glob.glob(self.train_image_dir + "/*." + self.image_suffix))
        self.train_mask_list = sorted(glob.glob(self.train_mask_dir + "/*." + self.image_suffix))

    def __len__(self):
        return len(self.train_mask_list)

    def transform(self, image, mask):
        """

        :param image: Stacked image eg (71, 1200, 1200)
        :param mask: Stacked mask eg (71, 1200, 1200)
        :return: Transformed images that have been randomly cropped, flipped, and resized to the target size.

        """
        resize_image = transforms.Resize(size=self.target_size)
        image = resize_image(image)
        mask = resize_image(mask)
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7

    
        # Random cropping

        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.target_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)



        return image, mask

    def __getitem__(self, img_index):
        """

        :param img_index: Index of the image
        :return: A dictionary containing an image, mask, and the index of that item.
        """
        if torch.is_tensor(img_index):
            img_index = img_index.tolist()
        if self.image_suffix == "tif":
            final_train_image = imread(self.train_image_list[img_index], plugin="pil")
            final_train_mask = imread(self.train_mask_list[img_index], plugin="pil")
        else:
            # WIP: Using uint16 with PIL leads to loss of information
            final_train_image = imread(self.train_image_list[img_index])
            final_train_mask = imread(self.train_mask_list[img_index])
        # Convert images to PIL/Tensor
        # List comprehension since we have image stacks.
        # uint16 doesn't work with resizing --> convert to uint8 but preserve information
        # Given the data, masks do not work well with alpha scaling
        final_mask_train = [Image.fromarray(convertScaleAbs(x)) for x in final_train_mask]
        final_image_train = [Image.fromarray(convertScaleAbs(x, alpha=255.0/65535.0)) for x in final_train_image]
        final_image_list = []
        final_mask_list = []

        for image, mask in zip(final_image_train, final_mask_train):
            image, mask = self.transform(image, mask)
            final_image_list.append(image)
            final_mask_list.append(mask)

        return {"image": final_image_list, "mask": final_mask_list, "index": img_index}
