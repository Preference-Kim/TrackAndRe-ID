import re
import os
from os.path import isfile, join

import cv2
import torch
from torchreid import metrics

class Batch:
    """simple data class that contain image lists for each id"""
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

# 다 만들면 utils에 넣고 테스트
class GalleryManager:
    """
        O; load gallery
        update gallery features
        matching id
        merging id
        managing gallery size
    """
    def __init__(self, model, buf_dir):
        self.buf_dir = buf_dir
        self.extractor = model
        self.features = {-1:torch.zeros(1,512).cuda()}

    def load_gallery(self):
        image_list = list_images_in_directory(self.buf_dir)
        batches = split_list_into_batches(image_list)

        for idx, batch in batches.items():
            self.features[idx]=self.extractor(batch())

    def update(self, im, id, cam, count):
        """update distinctive feature"""
        im_feat = self.extractor([im])
        if id in self.features:
            distmat = metrics.compute_distance_matrix(im_feat, self.features[id], metric='cosine')
            min_dist = torch.min(distmat).item()
            if min_dist>0.03:
                torch.cat((self.features[id], im_feat),dim=0)
                cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im)
        
    def sync_id(self, im, id, cam, count):
        """TODO: re-id inter camera"""
        im_feat = self.extractor([im])