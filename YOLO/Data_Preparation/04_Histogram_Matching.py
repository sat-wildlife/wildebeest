import cv2
import numpy as np
import os

def match_histogram_channel(channel, reference_channel):
    """
    Match the histogram of an image channel to the histogram of the reference channel.
    """
    channel_hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    channel_cdf = np.cumsum(channel_hist).astype(float)
    channel_cdf = 255 * channel_cdf / channel_cdf[-1]

    reference_hist, bins = np.histogram(reference_channel.flatten(), 256, [0, 256])
    reference_cdf = np.cumsum(reference_hist).astype(float)
    reference_cdf = 255 * reference_cdf / reference_cdf[-1]

    # Create a lookup table
    lookup_table = np.zeros(256)
    for i in range(256):
        lookup_table[i] = np.searchsorted(reference_cdf, channel_cdf[i])

    # Map the original image through the lookup table
    matched_channel = np.interp(channel.flatten(), bins[:-1], lookup_table)
    return matched_channel.reshape(channel.shape).astype(np.uint8)

def match_histograms_color(source_image, reference_image):
    """
    Match histograms of a color image to a reference color image.
    """
    matched_image = np.zeros(source_image.shape, dtype=np.uint8)
    for channel in range(3): # Assuming the image is in BGR format
        matched_image[:,:,channel] = match_histogram_channel(source_image[:,:,channel], reference_image[:,:,channel])
    return matched_image

def match_histograms_in_folder(folder_path, reference_image_path, output_folder_path):
    """
    Match histograms of all color images in a folder to a reference color image.
    """
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        raise ValueError("Reference image could not be read.")

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        source_image = cv2.imread(file_path)
        if source_image is None:
            print(f"Skipping {filename}, could not read image.")
            continue

        matched_image = match_histograms_color(source_image, reference_image)
        output_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_path, matched_image)
        print(f"Processed {filename}.")

source_folder = r''
reference_image_path = r''
output_folder = r''

match_histograms_in_folder(source_folder, reference_image_path, output_folder)


