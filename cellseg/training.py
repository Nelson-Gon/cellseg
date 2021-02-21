from torch.optim import Adam, RMSprop, SGD
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
train_images = DataProcessor("D:\\train_images\\images", target_size=(32, 32))
train_labels = DataProcessor("D:\\train_images\masks", dir_type="masks", target_size = (32, 32))
# Check that these are not uint16
test_dims = train_labels[0][0]
test_dims = test_dims.unsqueeze(0)
test_dims.size(-1)
test_dims.view(test_dims.size(0), -1).size()
train_images[0][0].dtype
# Train with each stacked image, each stacked image has 71 images
train_x = DataLoader(train_images[0], batch_size=2)
train_y = DataLoader(train_labels[0], batch_size=2)


# Define model
model = CellNet()
summary(model, input_size=(1,32, 32))
# Define Criterion
criterion = nn.BCELoss()
# Define optimizer
optimizer = Adam(model.parameters(), lr=0.001)

losses_per_batch = []

for epoch in range(1):
    for index, (img, msk) in enumerate(zip(train_images[0], train_labels[0])):
        # Add batch size
        img = img.unsqueeze(0)
        msk = msk.unsqueeze(0)
        output = model(img)
        print("out", output.size())
        print("img", img.size())

        loss = criterion(output, msk)
        losses_per_batch.append(loss)
        loss.backward()
        optimizer.step()
        print("Done*******************************")

