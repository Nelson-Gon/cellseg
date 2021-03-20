from torch.optim import Adam, RMSprop, SGD
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from .utils import DataProcessor
import os

dir_path = os.path.dirname((os.path.abspath(__file__)))

train_images = os.path.join(os.path.dirname(dir_path), "data/train/images")
train_labels = os.path.join(os.path.dirname(dir_path), "data/train/masks")
train_labels = DataProcessor(train_images, train_labels, target_size = (256, 256))
# # Check that these are not uint16
# test_dims = train_labels[0][0]
# test_dims = test_dims.unsqueeze(0)
# test_dims.size(-1)
# test_dims.view(test_dims.size(0), -1).size()
# train_images[0][0].dtype
# # Train with each stacked image, each stacked image has 71 images
# train_x = DataLoader(train_images[0], batch_size=2)
# train_y = DataLoader(train_labels[0], batch_size=2)
#
# train_x.dataset
#
# # Define model
# model = CellNet()
# summary(model, input_size=(1,32, 32))
# # Define Criterion
# criterion = nn.BCELoss()
# # Define optimizer
# optimizer = Adam(model.parameters(), lr=0.001)
#
# losses_per_batch = []
#
# for epoch in range(1):
#     for index, (img, msk) in enumerate(zip(train_images[0], train_labels[0])):
#         # Add batch size
#         img = img.unsqueeze(0)
#         msk = msk.unsqueeze(0)
#         output = model(img)
#         print("out", output.size())
#         print("img", img.size())
#
#         loss = criterion(output, msk)
#         losses_per_batch.append(loss)
#         loss.backward()
#         optimizer.step()
#         print("Done*******************************")

