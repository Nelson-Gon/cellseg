import unittest
import os
from cellseg.data import DataProcessor
from cellseg.utils import show_images
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

not_valid_path  = os.path.join(base_dir, "not_valid")

# Create a DataProcessor object from these paths

dataset_object = DataProcessor(image_dir=test_data_images, label_dir=test_data_labels, target_size=(512, 512),
                               image_suffix="tif")

class cellsegTests(unittest.TestCase):

    def test_instance_creation(self):

        self.assertIsInstance(dataset_object, DataProcessor)

    def test_DataProcessor_class(self):
        with self.assertRaises(NotADirectoryError) as err:
            DataProcessor(image_dir=not_valid_path,label_dir=test_data_labels)
        self.assertEqual(str(err.exception), "One or both directories do not exist.")

        # Ensure that we error on unequal lengths

        with self.assertRaises(ValueError) as err:
            DataProcessor(image_dir = test_data_images, label_dir= test_lengths,
                          image_suffix="tif")
        self.assertEqual(str(err.exception), "Found 10 images but 1 label.")

        # Ensure that we have the expected number of images

        self.assertEqual(dataset_object.__len__, 10)

        # Ensure we have a tuple in target size
        with self.assertRaises(TypeError) as err:
            DataProcessor(test_data_images,
                          test_data_labels,
                                  image_suffix="tif",
                                  target_size = [256, 256])
        self.assertEqual(str(err.exception), "Target size should be a tuple not list")

    def test_transforms(self):
        # Ensure that transforms work as expected
        transforms_object = DataProcessor(test_data_images,
                                          test_data_labels,
                                          target_size = (256, 256))

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


if __name__ == "__main__":
    unittest.main()