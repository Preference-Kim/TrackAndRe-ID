import os, re
from os.path import isfile, join
from pathlib import Path

import cv2
import numpy as np

from . import LOGGER

def get_pixel_params_mask(sources, vid_stride=3, count=1, threshold=(210, 210, 210)):
    # Convert sources to a list if it's not already
    sources = sources if isinstance(sources, list) else Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]

    pixel_means = []
    pixel_stds = []

    for s in sources:
        # Open RTSP streams
        s = eval(s) if s.isnumeric() else s
        cap = cv2.VideoCapture(s)

        n = 0

        while n < count:
            ret, frame = cap.read()

            if not ret:
                break

            if n % vid_stride == 0:
                # Create a mask for the current frame
                mask = cv2.inRange(frame, (0,0,0), threshold)
                # Get pixel values from the frame
                pixel_values = frame[np.where(mask > 0)].reshape(-1, 3)

                # Apply the mask to the pixel values
                active_pixels = pixel_values

                if len(active_pixels) > 0:
                    # Calculate mean and standard deviation using active pixels only
                    pixel_mean = np.mean(active_pixels, axis=0) / 255.0
                    pixel_std = np.std(active_pixels, axis=0) / 255.0
                else:
                    # Handle the case when there are no active pixels
                    pixel_mean = np.array([0, 0, 0])
                    pixel_std = np.array([0, 0, 0])

                pixel_means.append(pixel_mean)
                pixel_stds.append(pixel_std)

            n += 1

        # Close RTSP streams
        cap.release()

    # Calculate the overall mean and standard deviation
    overall_mean = np.mean(pixel_means, axis=0)
    overall_mean = np.around(overall_mean, 3).tolist()
    overall_std = np.mean(pixel_stds, axis=0)
    overall_std = np.around(overall_std, 3).tolist()

    LOGGER.info("")
    LOGGER.info(f"📷 Pixel Mean (Excluding Bright Areas):                {overall_mean}")
    LOGGER.info(f"📷 Pixel Standard Deviation (Excluding Bright Areas):  {overall_std}")

    return (overall_mean, overall_std)

class Batch:
    """Simple data class that contains image lists for each id."""
    def __init__(self, id, cam):
        self.id = id
        self.cam = cam
        self.batch = []
        self.feature = None
        
    def __call__(self):
        return self.batch

def list_images_in_directory(directory_path):
    """
    List all image files in the specified directory.

    :param directory_path: The path to the directory containing images.
    :return: A list of image file paths.
    """
    image_files = [join(directory_path, f) for f in os.listdir(directory_path) if isfile(join(directory_path, f))]

    # Filter the image files based on common image file extensions (you can customize this list).
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_files = [f for f in image_files if any(f.lower().endswith(ext) for ext in image_extensions)]

    return image_files

def split_list_into_batches(image_list):
    """
    Split a list of image files into batches based on 'id{index}_' part of the file names.
    
    :param image_list: A list of image file paths.
    :return: A dictionary where keys are 'index' values, and values are lists of file paths with the same 'id{index}_'.
    """
    batches = {}  # Dictionary to store batches
    for image_path in image_list:
        # Use regular expression to extract 'index' from the file name
        idx_match = re.search(r'id(\d+)_', image_path)
        cam_match = re.search(r'cam(\d+)', image_path)
        if idx_match:
            index = int(idx_match.group(1))  # Extract the 'index'
            if index in batches:
                batches[index]().append(image_path)
            else:
                batches[index] = Batch(id = index, cam = cam_match.group(1))
                batches[index]().append(image_path)
    
    return batches