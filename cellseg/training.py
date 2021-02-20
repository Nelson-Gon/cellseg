from torch.optim import Adam, RMSprop, SGD
from torch import nn
from torch.utils.data import DataLoader

train_images = DataProcessor("D:\\train_images\\images", target_size=(32, 32))
train_labels = DataProcessor("D:\\train_images\masks", dir_type="masks", target_size = (32, 32))
train_labels[0][0].dtype
train_images[0][0].dtype
# Train with each stacked image
train_x = DataLoader(train_images[0], batch_size=2)
train_y = DataLoader(train_labels[0], batch_size=2)


unsq= train_images[0][0].unsqueeze(0)

# Define model
model = CellNet()
# Define Criterion
criterion = nn.BCELoss()
# Define optimizer
optimizer = Adam(model.parameters(), lr=0.001)

losses_per_batch = []

for epoch in range(1):
    for index, (img, msk) in enumerate(zip(train_images[0], train_labels[0])):
        img = img.unsqueeze(0)
        msk = msk.unsqueeze(0)
        output = model(img)
        loss = criterion(output, msk)
        losses_per_batch.append(loss)
        loss.backward()
        optimizer.step()
        print("Done*******************************")

