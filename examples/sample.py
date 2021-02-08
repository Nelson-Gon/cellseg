from cellseg.cellseg.data import *
from cellseg.cellseg.utils import show_images
from torchvision.transforms import ToTensor


test = DataLoader("D:\\train_images", image_suffix="tif")
test[0]["image"]


