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

class ReIDManager:
    """
    Manage gallery features, matching id, and gallery size.
    """
    
    _active_ids = {-1:(-1,)} # list of Active IDs per camera {cam(int):ids(tuple of int)}
    _cam = {-1:-1} # id(int): cam(int)

    def __init__(self, model, buf_dir):
        self.buf_dir = buf_dir
        os.makedirs(self.buf_dir, exist_ok=True)
        self.extractor = model
        self.features = {-1: torch.zeros(1, 512).cuda()}
        self.min_dist_thres = 0.1
        self.max_dist_thres = 0.25
        # TODO State

    def load_gallery(self):
        image_list = list_images_in_directory(self.buf_dir)
        batches = split_list_into_batches(image_list)

        for idx, batch in batches.items():
            self.features[idx]=self.extractor(batch())

    def update(self, im, id, cam, count, issave = False):
        """update distinctive feature"""
        im_feat = self.extractor([im])
        reid = ReidMap.get_reid(id)
        if reid ==-1:
            fs = self.features.copy()
            if id in fs:
                distmat = metrics.compute_distance_matrix(im_feat, self.features[id], metric='cosine')
                if not (distmat<self.min_dist_thres).any():
                    self.features[id]=torch.cat((self.features[id], im_feat),dim=0)
                    cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im) if issave else None
            else:
                self.features[id] = im_feat
                cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im) if issave else None
        else:
            K = ReIDManager._cam.copy()
            for i in K:
                if i!=-1 and ReidMap.id_map.get(i,-1)==reid:
                    distmat = metrics.compute_distance_matrix(im_feat, self.features[id], metric='cosine')
                    if (distmat<self.min_dist_thres).any():
                        return
            self.features[id] = torch.cat((self.features[id], im_feat),dim=0)
            cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im) if issave else None
    
    def sync_id(self, id, cam):
        """
            TODO: re-id inter camera
            ids: list of active ids
            output: reid map dictionary
        """

        min_distance = 1

        gis = ReIDManager._active_ids.copy()
        for c, ids in gis.items(): #in self.features.keys() and active
            if cam == c:
                continue
            for i in ids:
                if i in self.features:
                    dismat = metrics.compute_distance_matrix(self.features[id], self.features[i], metric='cosine')
                    min_dismat = torch.min(dismat).item() 
                    # minimax_dismat = torch.max(dismat).item()

                    if min_dismat < min_distance:
                        min_distance = min_dismat
                        # minimax_distance = minimax_dismat
                        min_i = i
        
        if min_distance < self.max_dist_thres:
            ReidMap.map_reid(id, min_i)
    
    def remap_id(self, id):
        """
        TODO: re-id intra camera when self.features[id].shape[0]<30
        Args:
            id (_type_): _description_
            cam (_type_): _description_
        """
        fs=self.features.copy()
        
        min_distance = 1

        for i, f in fs.items():
            if i == id or i in ReIDManager._active_ids[ReIDManager._cam[i]]:
                continue
            Acts = ReIDManager._active_ids[ReIDManager._cam[id]]
            if ReidMap.get_reid(i) in [ReidMap.get_reid(ii) for ii in Acts]:
                continue
            dismat = metrics.compute_distance_matrix(self.features[id], f, metric='cosine')
            min_dismat = torch.min(dismat).item() 

            if min_dismat < min_distance:
                min_distance = min_dismat
                # minimax_distance = minimax_dismat
                min_i = i
        
        if min_distance < self.max_dist_thres:
            ReidMap.map_reid(id, min_i)
    
    @classmethod
    def list_actives(cls,ids,cam):
        cls._cam.update({i: cam for i in ids})        
        cls._active_ids[cam] = tuple(ids)
    
    @classmethod
    def get_actives(cls):
        return cls._active_ids.items()
        
    def _manage(self):
        """
        TODO: Capacity management.
        Inactive features may need to be removed.
        """
        pass
    
