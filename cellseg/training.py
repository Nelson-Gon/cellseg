# TODO: For each frame with an image stack, track cells, and quantify them
from torch.optim import Adam, RMSprop, SGD
from torch import nn

# Define Loss function --> BCE

loss_function = nn.BCELoss()
# Use Adam as optimizer
optimizer = Adam(model.parameters(), lr=0.0001)

# Track loss
losses_per_batch =[]

for epoch in range(3):
    get_iteration = iter(DataProcessor)
    for batch in get_iteration:
        optimizer.zero_grad()

        output = model(batch["image"])
        loss = loss_function(output, batch["masks"])
        losses_per_batch.append(loss)
        loss.backward()
        optimizer.step()
