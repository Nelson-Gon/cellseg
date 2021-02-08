from cellseg.cellseg import *


test = CellSeg("D:\\train_images", image_suffix="tif")

plt.imshow(test[0]["image"], cmap="gray")

# ===== Notes ======

# Get number of frames in the image

# tst_image.shape

# Read first frame
# tst_image[0]
# Plot and see what it looks like
# plt.imshow(tst_image[0], cmap="gray")

# Threshold
# thresholded = cv2.threshold(tst_image[0], 10, 255, cv2.THRESH_OTSU)
# thresholded[1].shape
# See after thresholding
# Ideas
# Track cells frame by frame ideally using a readily available model
# Modify model weights/layers
# Quantify each cell as training proceeds (just CV?)
# Classify as entering mitosis or not?
# Get threshold at which cell entered mitosis.
# Gaussian blur preprocessing increases over fitting risk? https://www.nature.com/articles/s41524-020-00363-x#Sec1