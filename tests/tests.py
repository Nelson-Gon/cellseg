import unittest
import os
from cellseg.data import DataProcessor
from cellseg.utils import show_images
from unittest import mock

# Create Paths to test files -- data here

# Get directory name one level up

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create paths from the above

test_data_images = os.path.join(base_dir, "data/train/images")
test_data_labels = os.path.join(base_dir, "data/train/masks")
# labels in another directory, use to ensure we error on incorrect
# Lengths of images and labels
test_lengths = os.path.join(base_dir, "data/validation/dummy_for_tests")

# Wrong paths

not_valid_path = os.path.join(base_dir, "not_valid")

# Create a DataProcessor object from these paths

dataset_object = DataProcessor(image_dir=test_data_images, label_dir=test_data_labels, target_size=(512, 512),
                               image_suffix="tif")


class TestClass(unittest.TestCase):

    def test_instance_creation(self):
        self.assertIsInstance(dataset_object, DataProcessor)

    def test_DataProcessor_class(self):
        with self.assertRaises(NotADirectoryError) as err:
            DataProcessor(image_dir=not_valid_path, label_dir=test_data_labels)
        self.assertEqual(str(err.exception), "One or both directories do not exist.")

        # Ensure that we error on unequal lengths

        with self.assertRaises(ValueError) as err:
            DataProcessor(image_dir=test_data_images, label_dir=test_lengths,
                          image_suffix="tif")
        self.assertEqual(str(err.exception), "Found 10 images but 1 label.")

        # Ensure that we have the expected number of images

        self.assertEqual(dataset_object.__len__, 10)

        # Ensure we have a tuple in target size
        with self.assertRaises(TypeError) as err:
            DataProcessor(test_data_images,
                          test_data_labels,
                          image_suffix="tif",
                          target_size=[256, 256])
        self.assertEqual(str(err.exception), "Target size should be a tuple not list")

    def test_transforms(self):
        # Ensure that transforms work as expected
        transforms_object = DataProcessor(test_data_images,
                                          test_data_labels,
                                          target_size=(256, 256))

        # Ensure that at each index we get images, labels, and an index

        first_index = next(iter(transforms_object))

        # Assert that this is actually a dict

        self.assertIsInstance(first_index, dict)

        # Check that both images and masks have had the same transformation

        first_index_image = first_index["image"]
        first_index_mask = first_index["mask"]
        self.assertEqual(first_index_image.shape[-1], 256)
        self.assertEqual(first_index_mask.shape[-1], 256)
        self.assertEqual(first_index["index"], 0)

    def test_utils(self):
        with self.assertRaises(ValueError) as err:
            show_images(dataset_object, number=42)
        self.assertEqual(str(err.exception), "number should be a non-zero int and less than or equal to 10 not 42")
        # Ensure that we only have image and mask in target
        # There must be a way to not repeat these steps and test at once
        with self.assertRaises(ValueError) as err:
            show_images(dataset_object, number=8, target="gibberish")

        self.assertEqual(str(err.exception), "Target should be one of image or mask not gibberish")

        # Check that we provide the right data type
        with self.assertRaises(TypeError) as err:
            show_images("not a DataProcessor object", number=4)

        self.assertEqual(str(err.exception), "Expected an object of class DataProcessor not str")

    # Mock plots
    @mock.patch("cellseg.utils.plt")
    def test_plots(self, mock_plt):
        show_images(dataset_object, number=4)
        mock_plt.cmap.called_once_with("gray")
        mock_plt.figure.called_once()
        # mock_plt.figure().add_subplot.assert_called_once()


if __name__ == "__main__":
    unittest.main()
