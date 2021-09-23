# from torch.optim import Adam, RMSprop, SGD
# from torch import nn
# from torch.utils.data import DataLoader
# from torchsummary import summary
from .utils import DataProcessor, show_images
import os
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image-directory", type=str, required=True,
                        help="Path to image directory containing images and masks/labels")
    parser.add_argument("-s", "--image-size", type=int, required=True, help="Size of images", default=256)
    parser.add_argument("-t", "--target", type=str, required=True, help="Target images to show", default="image")
    parser.add_argument("-n", "--number", type=int, required=True, help="Number of images to show", default=4)
    actual_args = parser.parse_args()
    train_images = os.path.join(actual_args.image_directory, "images")
    train_labels = os.path.join(actual_args.image_directory, "masks")
    train_labels = DataProcessor(train_images, train_labels, target_size=(actual_args.image_size,
                                                                          actual_args.image_size))
    # Show images
    show_images(train_labels, target=actual_args.target, number=actual_args.number)
    plt.show()
    # # Check that these are not uint16
    # test_dims = train_labels[0][0]
    # test_dims = test_dims.unsqueeze(0)
    # test_dims.size(-1)
    # test_dims.view(test_dims.size(0), -1).size()
    # train_images["image"][0].dtype
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


if __name__ == "__main__":
    main()
