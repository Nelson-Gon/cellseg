import torch.nn as nn


class CellNet(nn.Module):
    def __init__(self, input_shape=32, channels=1):
        super(CellNet, self).__init__()
        # in_channels: 1 for gray, 3 for rgb
        # out_channels: Depends on model architecture
        self.input_shape = input_shape
        self.channels = channels
        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.input_shape, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define length of flattened layer
        # Calculating input of FCN w' = (w - f + 2p)/s + 1
        # After halving with max pooling and same padding --> [batch_size, 32, 30] --> 30 /2 --> 15
        self.fc1 = nn.Linear(32 * 15 * 15, 64)  # Dense layer with output 64
        self.drop = nn.Dropout(0.5)

        # input from previous layer
        self.out = nn.Linear(64, 1) # number of classes based on input
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))  # [batch_size, 32, 32, 30]
        x = self.pool(x)  # [batch_size, 32, 15, 15]
        print(x.size())

        x = x.view(x.size(0), -1)  # [batch_size, 32*15*15=7200] #number of input features
        print(x.size())
        x = self.act(self.fc1(x))  # [batch_size, 64]
        print(x.size())
        x = self.drop(x)
        print(x.size())
        x = self.out(x)  # [batch_size, number_predictions]
        return x
