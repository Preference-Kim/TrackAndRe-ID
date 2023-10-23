import re
import os
from os.path import isfile, join

import cv2
import torch
from torchreid import metrics

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

class ReidMap:
    _count = 0
    id_map = {}  # Class-level dictionary to store id-reid mappings

    def __init__(self):
        pass

    @classmethod
    def map_reid(cls, id, id_dst):
        """
        Map reid from id to id_dst.

        Args:
            id (int): Source id.
            id_dst (int): Destination id.
        """
        if id_dst in cls.id_map:
            reid = cls.id_map[id_dst]
        else:
            cls._count += 1
            reid = cls._count
            cls.id_map[id_dst] = reid

        cls.id_map[id] = reid

    @classmethod
    def get_reid(cls, id):
        """
        Get the reid for a given id.

        Args:
            id (int): The id to retrieve reid for.

        Returns:
            int: The reid for the given id.
        """
        if id in cls.id_map:
            return cls.id_map[id]
        else:
            return -1  # Dummy id    

class GalleryManager:
    """
    Manage gallery features, matching id, and gallery size.
    """
    
    active_ids = {} # list of Active IDs per camera

    def __init__(self, model, buf_dir):
        self.buf_dir = buf_dir
        self.extractor = model
        self.features = {}
        self.cam = {}
        # TODO State

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
        else:
            self.cam[id] = cam
            self.features[id] = im_feat
            cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im)
    
    def sync_id(self, ids, id, cam):
        """
            TODO: re-id inter camera
            ids: list of active ids
            output: reid map dictionary
        """

        min_distance = 1

        for i in ids: #in self.features.keys() and active
            if i != id and self.cam[i]!= cam:
                dismat = metrics.compute_distance_matrix(self.features[id], self.features[i], metric='cosine')
                min_dismat = torch.min(dismat).item()
                # minimax_dismat = torch.max(dismat).item()

                if min_dismat < min_distance:
                    min_distance = min_dismat
                    # minimax_distance = minimax_dismat
                    min_i = i
        
        if min_distance < .25:
            ReidMap.map_reid(id, min_i)
    
    @classmethod
    def list_actives(cls,ids,cam):
        cls.active_ids[cam]=ids
    
    def remap_id(self, id, cam):
        """
        TODO: re-id intra camera
        Args:
            id (_type_): _description_
            cam (_type_): _description_
        """
        pass
        
    def _manage(self):
        """
        TODO: Capacity management.
        Inactive features may need to be removed.
        """
        pass
