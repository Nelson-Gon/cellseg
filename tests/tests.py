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


if __name__ == "__main__":
    unittest.main()