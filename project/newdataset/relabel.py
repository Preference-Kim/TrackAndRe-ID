import sys
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID")
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID/mylibrary")
import os
from os.path import isfile, join
import re

import cv2

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

class Batch:
    def __init__(self, id, cam):
        self.id = id
        self.cam = cam
        self.batch = []
        
    def __call__(self):
        return self.batch

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

id_mapping = {
    1: [1, 108],  # junaid
    2: [2, 66, 133],  # pradyumna
    3: [3],  # 관주
    4: [4, 12, 23, 30, 58, 106, 131],  # tu
    5: [5, 85, 145],  # steve
    6: [6, 13, 81],  # turing
    7: [7, 15, 35, 37, 45, 50, 79, 103, 116, 128, 142, 144, 153, 176, 180],  # maksym
    8: [8],  # umair
    9: [9, 38],  # jiang
    10: [14, 25, 28],  # rabia
    11: [17, 78, 80, 84, 92, 98, 123, 136, 182],  # sebastian
    12: [21, 27, 77, 95],  # edward
    13: [29],  # 명우
    14: [32, 40, 52, 110, 117, 157, 162],  # zahid
    15: [33, 46, 65, 90, 149, 174],  # 선호
    16: [34],  # hamza
    17: [39, 100],  # sajid
    0: [41, 91, 121],  # mohamed
}

def find_reid(old_id):
    
    # 새로운 ID로 매핑
    new_id = None
    for reid_id, id_list in id_mapping.items():
        if old_id in id_list:
            new_id = reid_id
            break
    return new_id

if __name__ == '__main__':
    # Define the source directory containing images
    directory_path = '/home/sunhokim/Documents/mygit/TrackAndRe-ID/images/buf-l-2023-10-19-1'
    
    # List all the image files in the source directory
    im_list = list_images_in_directory(directory_path)
    batches = split_list_into_batches(im_list)
    
    # Create an output directory for saving processed images
    out_dir = '/home/sunhokim/Documents/mygit/TrackAndRe-ID/images/dataset-buf-l'
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize a dictionary to keep track of reid values
    reid_stack = {}
    
    # Iterate through batches of images
    for id in batches.keys():
        reid = find_reid(batches[id].id)  # Find the reid value for the batch
        cam = batches[id].cam  # Get the camera ID
        if reid not in reid_stack:
            reid_stack[reid] = 0  # Initialize reid counter if not exists
        
        for im in batches[id]():
            # Read each image from the batch
            frame = cv2.imread(im)
            
            # Define the output file name based on reid, cam, and reid counter
            file_name = f'pid{reid}_camid{cam}_{reid_stack[reid]}.jpg'
            
            # Save the image to the output directory
            file_path = os.path.join(out_dir, file_name)
            cv2.imwrite(file_path, frame)
            
            # Increment the reid counter
            reid_stack[reid] += 1
    
    print('done')