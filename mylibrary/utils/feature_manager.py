import os
from threading import Thread

import cv2
import torch

from . import MyQueue
from .loader_util import list_images_in_directory, split_list_into_batches
from torchreid import metrics
class Features:
    fs = {-1: torch.zeros(1, 512).cuda()} # ids(int): featurematrix(torch.Tensor)
    
    @classmethod
    def __call__(cls):
        return cls.fs

class ReIDentify:
    """
    Manage gallery features, matching id, and gallery size.
    """
    
    _active_ids = {-1:(-1,)} # list of Active IDs per camera {cam(int):ids(tuple of int)}
    _cam = {-1:-1} # id(int): cam(int)
    min_dist = 0.1
    max_dist = 0.25

    def __init__(self, model, buf_dir):
        self.buf_dir = buf_dir
        os.makedirs(self.buf_dir, exist_ok=True)
        #self.features = Features()
        self.extractor = model
        # TODO State

    @staticmethod
    def set_thretholds(mind=0.1, maxd=0.25):
        ReIDentify.min_dist = mind
        ReIDentify.max_dist = maxd

    def load_gallery(self):
        """
        TODO: need to revise
        """
        image_list = list_images_in_directory(self.buf_dir)
        batches = split_list_into_batches(image_list)

        for idx, batch in batches.items():
            Features.fs[idx]=self.extractor(batch())

    def ft_update_reid(self, cam, count, im, id, reid, issave = False):
        im_ft = self.extractor([im])
        is_wrong = True # initialization
        for r in IDManager.reid2id[reid].copy():
            fts = Features()[r].clone().detach()
            dist_mat = metrics.compute_distance_matrix(im_ft, fts, metric='cosine')
            if (dist_mat<ReIDentify.min_dist).any():
                return
            else:
                is_wrong &= (dist_mat>ReIDentify.max_dist).all()
        if is_wrong: # wrong reid, need to correct
            IDManager.reset_reid(id, reid)
            Features.fs[id] = im_ft           
        else: # good feature, need to add
            Features.fs[id] = torch.cat((Features.fs[id], im_ft),dim=0)
        cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im) if issave else None

    def ft_update_id(self, cam, count, im, id, issave = False):
        im_ft = self.extractor([im])
        if id not in Features():
            Features.fs[id] = im_ft
        else:
            fts = Features()[id].clone().detach()
            dist_mat = metrics.compute_distance_matrix(im_ft, fts, metric='cosine')
            if (dist_mat>ReIDentify.max_dist).all(): # supposed that id is switched, need to correct
                Features.fs[id] = im_ft
            elif (dist_mat>ReIDentify.min_dist).all():
                Features.fs[id] = torch.cat((Features.fs[id], im_ft),dim=0)
            else:
                return
        cv2.imwrite(f'{self.buf_dir}/id{id}_cam{cam}_{count}.jpg', im) if issave else None

    @staticmethod
    def sync(id):
        if id in IDManager.synced_ids_REMOVED.values(): #################TODO:
            return
        cam = IDManager.i2c[id]
        c2i = IDManager.active_c2i.copy()
        blacklist = set()
        while True:
            min_d = 1
            nearest_id = None
            for c, ids in c2i.items():
                if c == cam:
                    continue
                for i in ids - blacklist:
                    if i in Features():
                        dismat = metrics.compute_distance_matrix(Features.fs[id], Features.fs[i], metric='cosine')
                        distance = torch.min(dismat).item()
                        if distance > ReIDentify.max_dist:
                            blacklist.add(i)
                            continue
                        elif distance < min_d:
                            min_d = distance
                            nearest_id = i
            if nearest_id:
                if nearest_id in IDManager.synced_ids_REMOVED:              # 이미 누가 따라가고 있다면?
                    (challenger, record) = IDManager.synced_ids_REMOVED[nearest_id]
                    if challenger not in IDManager.active_c2i[IDManager.i2c[challenger]]:
                        IDManager.synced_ids_REMOVED[nearest_id] = (id, min_d) 
                        break    
                    elif min_d > record:
                        blacklist.add(nearest_id)
                    else:
                        IDManager.synced_ids_REMOVED[nearest_id] = (id, min_d) 
                        break # TODO: reid 처음에 id2reid, synced_ids로 재맵핑하기
            else:
                break
    
    def remap_id(self, id):
        """
        TODO: re-id intra camera when features[id].shape[0]<30
        Args:
            id (_type_): _description_
            cam (_type_): _description_
        """
        fs=Features.fs.copy()
        
        min_distance = 1

        for i, f in fs.items():
            if i == id or i in ReIDentify._active_ids[ReIDentify._cam[i]]:
                continue
            Acts = ReIDentify._active_ids[ReIDentify._cam[id]]
            if ReidMap.get_reid(i) in [ReidMap.get_reid(ii) for ii in Acts]:
                continue
            dismat = metrics.compute_distance_matrix(Features.fs[id], f, metric='cosine')
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

class IDManager: 

    def __init__(self, num_cam = None, gallery_life = 60):
        
        """Queue for get tasks"""
        IDManager.task = set(IDManager.task_map().keys())
        IDManager.task_q = MyQueue(maxsize=0, 
                                   task=IDManager.task
                                   )                                    # queue for id(live track/reid/sync) update, sync, reid
        IDManager.num_cam = num_cam   
        """Track ID"""
        IDManager.i2c = {}                                              # dict - i2c[id] = cam
        # active
        IDManager.active_c2i = [set() for _ in range(num_cam)]          # list - active_c2i[cam] = {id, ...} (set)
        IDManager.newest_id = [0 for _ in range(num_cam)]               # list - newest_id[cam] = id (int)
        # vanished
        IDManager.missing_c2i = [([],[]) for _ in range(num_cam)]       # list - missing_c2i[cam] = ( [id (int), ...], [age (int), ...] ) (tuple) # BEFORE: {did (int): score (int)} 
        # gallery
        IDManager.gallery = [([],[]) for _ in range(num_cam)]           # list - gallery[cam] = ( [id (int), ...], [age (int), ...] ) (tuple), all ids have reid
        IDManager.gallery_life = gallery_life
        """Sync"""
        IDManager.synced_sets = []                                      # list - synced_sets[idx] = ([reid], [id]*num_cam, [distance]*num_cam) (tuple)
        IDManager.synced_ids_REMOVED = {}                               # TODO: need to remove(asymmertic). synced_ids[followee] = (follower(int), distance(float)) (tuple)
        """Re ID"""
        IDManager._new_reid = 0                                         # int
        IDManager.id2reid = {}                                          # dict - id2reid[id] = reid (int)
        IDManager.reid2id = {}                                          # dict - reid2id[reid] = set([id, ...]) (set)

    @classmethod
    def settings(cls, num_cam):
        cls.num_cam = num_cam
        # Track ID: active
        cls.active_c2i = [set() for _ in range(num_cam)]    # {id, ...} (set)
        cls.newest_id = [0 for _ in range(num_cam)]         # id (int)
        # Track ID: vanished
        cls.missing_c2i = [([],[]) for _ in range(num_cam)] # ( [id (int), ...], [age (int), ...] ) (tuple) # BEFORE: {did (int): score (int)} (dict)
        # Track ID: gallery
        cls.gallery = [([],[]) for _ in range(num_cam)]     # ( [id (int), ...], [age (int), ...] ) (tuple)

    @classmethod
    @property
    def task_map(cls):
        return {
            'update': cls.id_update,
        }
    
    @classmethod
    def work(cls, running):
        while running():
            task, items = cls.task_q.get_query(timeout=2)
            if not items:                       # is None
                cls.task_map[task]()
            elif isinstance(items, tuple):      # is positional arguments
                cls.task_map[task](*items)
            else:                               # isinstance(items, dict)
                cls.task_map[task](**items)
    
    @classmethod
    def id_update(cls, cam, active_ids, removed_ids):
        """
        1. Update the list of currently active IDs
            if an active id is new:
                update (active_c2i, i2c)
                sync set에 age=1이랑 같이 넣기
            else:
                update active_c2i
        2. Check for removed IDs and determine whether to save them in the gallery
            synced_sets에서 얘 빼기, 다 비어있는 set은 제거하기
            if an removed id is reided:
                gallery에 넣기
            else:
                버리기
        3. Increase ages for synced set and Checkup each to determine whether to give new reid

        Args:
            cam (int): index of input channel
            active_ids (list of int): list of current ids in cam
            removed_ids (set of int): list of removed ids from ByteTrack
            
        Referenced class variables:
            active_c2i
            newest_id
            i2c
            missing_c2i
            gallery
            gallery_life
            synced_sets
            
        Called methods:
            cls.get_reid(id) # to check for removed ID    
            TODO: checkupreid
        """
        
        """1. Update the list of currently active IDs"""
        newest_id = 0
        cls.active_c2i[cam] = set(active_ids)
        for id in active_ids:
            if id > cls.newest_id[cam]: # If an active id is new
                newest_id = max(newest_id, id)
                cls.i2c[id] = cam
                new_set = (
                            [],                                   # Similarity with original reids (for positive, it is reid. for negative, it is similarity)
                            [None for _ in range(cls.num_cam)],   # IDs
                            [None for _ in range(cls.num_cam)]    # Distances
                           )
                new_set[0][:] = [-1] # it will decrease if this set keeps unassigned state
                new_set[1][cam] = id
                new_set[2][cam] = None # TODO: 다른 id가 여기 set에 추가될 때 nearest id의 distance도 같이 바꿔줘야할듯? 
                IDManager.synced_sets.append(new_set)
            else:
                continue
        cls.newest_id[cam] = max(cls.newest_id[cam], newest_id)
        
        """2. Check for removed IDs and determine whether to save them in the gallery"""
        for idx, (property, ids, dists) in enumerate(cls.synced_sets.copy()):
            for id in removed_ids.copy():
                if id in ids:
                    removed_ids.remove(id)
                    ids[cam] = None # cam = ids.index(id)
                    dists[cam] = None
                    if int(*property)>0: # If the removed id has reid
                        cls.add_new_asset(cam, id) # Add to gallery
            if not any(ids): # If the set get empty
                del(cls.synced_sets[idx])
                
        """TODO: 3. Increase ages for synced set and Checkup each to determine whether to give new reid"""
            
            
        
        """Originals"""
        
        cls.active_c2i[cam] = set(active_ids)
        for id in active_ids:
            if id > cls.newest_id[cam]:
                cls.i2c[id] = cam              
                cls.newest_id[cam] = id
            reid = cls.get_reid(id)
    
    @staticmethod
    def add_new_asset(cam, id):
        """newly add an id to list of gallery

        Args:
            cam (int): index of input channel
            id (int): track id
        """
        if IDManager.get_reid(id): # if it's unidentified ID, it will be thrown away
            if IDManager.gallery[cam]:
                IDManager.gallery[cam].append([id, 0])
            else:
                IDManager.gallery[cam] = [[id, 0]]
        else:
            del(Features.fs[id])

    @staticmethod
    def checkup_assets(cam, fps):
        """check if there exists any out-of-date id in gallery

        Args:
            cam (int): index of input channel
            fps (float): fps of input channel
        """
        if not IDManager.gallery[cam]:
            return
        else:
            for i, asset in enumerate(IDManager.gallery[cam]):
                asset[-1] += 1
                if asset[-1] > IDManager.gallery_life*fps:
                    IDManager.reset_reid(asset[0], None)
                    del(IDManager.gallery[cam][i])
                    del(Features.fs[asset[0]])
    
    @staticmethod
    def get_reid(id):
        """return reid of id

        Args:
            id (int): track id

        Returns:
            if id is mapped:
                reid (int)
            else:
                None
        """
        if id in IDManager.id2reid:
            return IDManager.id2reid[id]
        else:
            return None

    @staticmethod
    def map_reid(id, id_dst):
        """
        Map reid from id to id_dst.

        Args:
            id (int): Source track id.
            id_dst (int): Destination id.
        """
        if id_dst in IDManager.id2reid:
            reid = IDManager.id2reid[id_dst]
        else:
            IDManager._new_reid += 1
            reid = IDManager._new_reid
            IDManager.id2reid[id_dst] = reid
            IDManager.reid2id[reid] = {id_dst}
        IDManager.id2reid[id] = reid
        IDManager.reid2id[reid].add(id)
    
    @staticmethod
    def reset_reid(id, reid):
        del(IDManager.id2reid[id])
        if reid:
            IDManager.reid2id[reid].discard(id)
        else:
            r = IDManager.get_reid(id)
            IDManager.reid2id[r].discard(id)
    
    """
    
    -- Synchronization Methods --
    
    """
    
    @staticmethod
    def get_synced_set(id):
        # synced_sets = []    # list of [reid, [id]*num_cam, [distance]*num_cam]
        sets = IDManager.synced_sets
        for idx_set, (reid, ids, distances) in enumerate(sets): # set[0] = reid(int), set[1], 
            if id in ids:
                return True, (idx_set, ids.index(id), reid, ids, distances)
        return False, None

    @staticmethod
    def remove ############TODO