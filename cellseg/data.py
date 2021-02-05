from torch.utils.data import Dataset
import pandas as pd
import cv2
from pandas import json_normalize
import os
import torch
from pyautocv.segmentation import show_images


class DataClass(Dataset):
    def __init__(self, image_dir, annotations_path):
        """
        This class inherits from torch.Dataset to customize image and annotation loading


        """

        self.annotations_path = pd.read_csv(annotations_path)

        self.image_dir = image_dir

    def __len__(self):
        return len(self.annotations_path.iloc[:, 0])

    def __getitem__(self, img_index):
        """
        This allows us to load each image and its annotation by index
        """
        if torch.is_tensor(img_index):
            img_index = img_index.tolist()

        image_name = os.path.join(self.image_dir, self.annotations_path.iloc[img_index, 0])

        actual_image = cv2.imread(image_name)
        features = self.annotations_path.iloc[img_index, 5]
        # convert features to DataFrame from Series

        features = pd.DataFrame(pd.read_json(features, typ="series", orient="records")).transpose()

        # Get specific features
        x_points = features["cx"]
        y_points = features["cy"]

        features_image = {"image": actual_image, "features": features, "x": x_points, "y": y_points}
        return features_image

    # TODO: Figure out how to write a plotting method --> use show_images from pyautocv for now


