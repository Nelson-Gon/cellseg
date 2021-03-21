# DataProcessor
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from skimage.io import imread
from PIL import Image
import torch
import glob
import random
import os



class DataProcessor(Dataset):
    def __init__(self, image_dir, label_dir,target_size=(512, 512), image_suffix="tif"):
        """

        :param image_dir: Path to image directory.
        :param label_dir: Path to labels/masks
        :param target_size: Target size to resize to. Defaults to (512, 512). Useful if you have low memory or
        computational power.
        :param image_suffix: Image suffix of the images. Defaults to "tif"

        :return An object of class DataProcessor that inherits from torch.utils.data.Dataset

        """
        # Initial idea was to read image stacks
        # This has since changed. Therefore need to update changes
        # Read and process images and masks together
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_suffix = image_suffix
        self.target_size = target_size




        # Assert that directories actually exist
        if not all(os.path.isdir(x) for x in (self.label_dir, self.image_dir)):
            raise NotADirectoryError("One or both directories do not exist.")

        # raise typeerror at instance creation
        if not isinstance(self.target_size, tuple):
            raise TypeError(f"Target size should be a tuple not {type(self.target_size).__name__}")

        self.image_list = sorted(glob.glob(self.image_dir + "/*." + self.image_suffix))
        self.label_list = sorted(glob.glob(self.label_dir + "/*." + self.image_suffix))

        if len(self.image_list) != len(self.label_list):
            no_images = "image" if len(self.image_list) == 1 else "images"
            no_labels = "label" if len(self.label_list) == 1 else "labels"
            raise ValueError(f"Found {len(self.image_list)} {no_images} but {len(self.label_list)} {no_labels}.")


    @property
    def __len__(self):
        """
        :return: Number of images found.
        """
        return len(self.image_list)

    def transform(self, image, mask):
        """

        :param image: Image to be transformed.
        :param mask:  Label/Mask to be transformed.
        :return: Transformed images that have been randomly cropped, flipped, and resized to the target size.

        """
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7



        resize_image = transforms.Resize(size=self.target_size)
        image = resize_image(image)
        mask = resize_image(mask)



        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.target_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)


        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)


        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask



    def __getitem__(self, img_index):
        """

        :param img_index: Index of the image
        :return: A dict containing an image, its label, and index.
        """

        if torch.is_tensor(img_index):
            img_index = img_index.tolist()

        use_plugin = "pil" if self.image_suffix == "tif" else None

        final_image = imread(self.image_list[img_index], plugin=use_plugin)
        final_label = imread(self.label_list[img_index], plugin=use_plugin)

        image, mask = self.transform(Image.fromarray(final_image), Image.fromarray(final_label))
        return {"image": image, "mask": mask, "index": img_index}




