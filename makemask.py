import cv2
import numpy as np
import os

# Directory containing the input images
input_dir = 'dataset/GroundTruthImages/'

# Directory to save the masked images
maskimage_dir = 'dataset/InputImages/'
mask_dir = 'dataset/Mask/'

# Create the output directory if it doesn't exist
os.makedirs(maskimage_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

# Get a list of all JPG files in the input directory
file_list = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

# Iterate over each file in the directory
for file_name in file_list:
    # Read input image
    img = cv2.imread(os.path.join(input_dir, file_name))

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[0:2500, 0:450] = 255

    # Compute the bitwise AND using the mask
    # masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Save the masked image in the output directory
    # maskimage_path = os.path.join(maskimage_dir, file_name)
    mask_path = os.path.join(mask_dir, file_name)

    # cv2.imwrite(maskimage_path, masked_img)
    cv2.imwrite(mask_path, mask)