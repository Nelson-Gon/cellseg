import torch.nn as nn
import torch


class CellNet(nn.Module):
    def __init__(self):
        super(CellNet, self).__init__()
        # in_channels: 1 for gray, 3 for rgb
        # out_channels: Depends on model architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, out_channels=32, kernel_size=3)
        # Define length of flattened layer
        # Calculating input of FCN w' = (w - f + 2p)/s + 1 where w is from inputs eg 1 * 16 * 16
        # For a 4 by 4 kernel
        self.fc1 = nn.Linear(4 * 4 * 32, 100)
        # input from previous layer
        self.fc2 = nn.Linear(in_features=100, out_features=50)
        # Input from previous FCN, output number of predictions
        self.out = nn.Linear(50, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, kernel_size=3, stride=1)

        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = torch.max_pool2d(x, kernel_size=3, stride=1)

        x = x.view(-1, 4 * 4 * 32)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


test = CellNet()
print(test)