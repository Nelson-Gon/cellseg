from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from skimage.io import imread
from PIL import Image
import numpy
import torch
import glob
import random


class DataProcessor(Dataset):
    def __init__(self, train_image_dir, train_mask_dir, target_size=(130, 130), image_suffix="tif"):
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

        :param images_list: Stacked image eg (71, 1200, 1200)
        :param masks_list: Stacked mask eg (71, 1200, 1200)
        :return:

        """
        resize_image = transforms.Resize(size=self.target_size)
        image = resize_image(image)
        mask = resize_image(mask)
        # https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
        """
    
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
        """
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)



        return image, mask

    def __getitem__(self, img_index):
        if torch.is_tensor(img_index):
            img_index = img_index.tolist()
        if self.image_suffix == "tif":
            final_train_image = Image.open(self.train_image_list[img_index])
            final_train_mask = Image.open(self.train_mask_list[img_index])
        else:
            # WIP: Using uint16 with PIL leads to loss of information
            final_train_image = imread(self.train_image_list[img_index])
            final_train_mask = imread(self.train_mask_list[img_index])
        # Convert images to PIL/Tensor
        # List comprehension since we have image stacks.
        # uint8 distorts images but uint16 doesn't work with resizing
        # [Image.fromarray(x.astype("uint16")) for x in final_mask_train]
        #final_train_mask = [Image.fromarray(x) for x in final_mask_train]
        #final_train_image = [Image.fromarray(x) for x in final_image_train]
        final_image_list = []
        final_mask_list = []





        for image, mask in zip(final_train_image, final_train_mask):
            image, mask = self.transform(image, mask)
            final_image_list.append(image)
            final_mask_list.append(mask)

        return {"image": final_image_list, "mask": final_mask_list, "index": img_index}
