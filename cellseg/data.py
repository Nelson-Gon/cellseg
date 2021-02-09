from torch.utils.data import Dataset
from skimage.io import imread
import torch
import glob


class DataProcessor(Dataset):
    def __init__(self, image_directory, image_suffix="tif"):
        self.image_directory = image_directory
        self.image_suffix = image_suffix
        self.image_list = sorted(glob.glob(self.image_directory + "/*." + self.image_suffix))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, img_index):
        if torch.is_tensor(img_index):
            img_index = img_index.tolist()
        if self.image_suffix == "tif":
            final_image = imread(self.image_list[img_index], plugin="pil")
        else:
            final_image = imread(self.image_list[img_index])

        return {"image": final_image, "index": img_index}



