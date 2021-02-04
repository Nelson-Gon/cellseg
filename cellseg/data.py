from torch.utils.data import Dataset
import pandas as pd
import cv2
from pandas import json_normalize
import os
import torch


class DataClass(Dataset):
    def __init__(self, image_dir, annotations_path):
        """
        This class inherits from torch.Dataset to customize image and annotation loading

        """

        self.annotations_path = pd.read_json(annotations_path, orient="index")
        self.annotations_path.reset_index(inplace=True, drop=True)
        self.image_dir = image_dir

    def __getitem__(self, img_index):
        """
        This allows us to load each image and its annotation by index
        """
        if torch.is_tensor(img_index):
            img_index = img_index.tolist()

        img_name = os.path.join(self.image_dir, self.annotations_path.iloc[img_index, 0])
        image = cv2.imread(img_name)
        features = json_normalize(self.annotations_path.iloc[img_index, 2])
        # x points
        x_points = features["shape_attributes.all_points_x"].iloc[img_index]
        y_points = features["shape_attributes.all_points_y"].iloc[img_index]
        phase = features["region_attributes.phase"].iloc[img_index]
        # Read features as json

        features_image = {"image": image, "features": features, "x": x_points, "y": y_points, "phase": phase}
        return features_image

   