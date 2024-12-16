import os
import gdal
import numpy as np
from PIL import Image

# Define source folder and target folder
source_folder = r''  # Path to the folder containing the tif images
target_folder = r''  # Target folder to save PNG sub-images

# Create the target folder (if it doesn't exist)
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Read all tif files in the source folder
tif_files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]

# Define the size of sub-images
sub_image_width = 336
sub_image_height = 336

# Extract 100 sub-images from each tif image
for tif_file in tif_files:
    file_path = os.path.join(source_folder, tif_file)
    dataset = gdal.Open(file_path)
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize

    # Check the number of channels to ensure the image has at least 3 channels
    if dataset.RasterCount < 3:
        print(f"Image {tif_file} has less than 3 channels, skipping...")
        continue

    # Determine the starting points for random sub-images
    np.random.seed(0)  # Set random seed for reproducibility
    x_offsets = np.random.randint(0, xsize - sub_image_width, 400)
    y_offsets = np.random.randint(0, ysize - sub_image_height, 400)

    for i, (x_offset, y_offset) in enumerate(zip(x_offsets, y_offsets)):
        # Read sub-images for all three channels
        sub_images = [
            dataset.GetRasterBand(j + 1).ReadAsArray(int(x_offset), int(y_offset), sub_image_width, sub_image_height)
            for j in range(3)
        ]

        # Stack the sub-images into a three-channel image
        sub_image = np.dstack(sub_images)

        # Save as PNG
        sub_image_path = os.path.join(target_folder, f'{os.path.splitext(tif_file)[0]}_sub_{i}.png')
        Image.fromarray(sub_image).save(sub_image_path)

print("Sub-image extraction and saving completed!")
