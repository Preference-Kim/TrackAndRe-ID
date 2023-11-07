from threading import Thread
import queue
from copy import deepcopy

from .feature_manager import ReidMap, IDManager, Features

from . import LOGGER

class ReIDThread(Thread):
    def __init__(self, streams, camid, feature_man, queue, stepsize=1, issave=False):
        """_summary_

        Args:
            streams (_type_): _description_
            camid (_type_): _description_
            feature_man (_type_): _description_
            queue (_type_): _description_
            issave (bool): _description_
                camid = i,
                streams=streams,
                feature_man = reid_mans[i],
                queue = t_track.reid_queue,
                stepsize = stepsize,
                issave = is_save  
        """
        super(ReIDThread, self).__init__()
        self.cam = camid
        self.running = streams.running
        self.stepsize = stepsize
        self.queue = queue
        self.manager = feature_man
        self.issave = issave
        self.daemon = True

    def run(self):
        while self.running():
            try:
                (frame, count, boxes, indices) = self.queue.get(timeout=2)
                modc = count % self.stepsize
                for index, box in zip(indices, boxes):
                    reid = IDManager.get_reid(index)
                    modi = index % self.stepsize
                    if modc != modi:
                        continue
                    x1, y1, x2, y2 = box
                    cropped = frame[y1:y2, x1:x2]
                    if reid:
                        self.manager.update_reid(cam = self.cam, count = count, im = cropped, id = index, reid = reid, issave = self.issave)
                    else:
                        self.manager.update_id(cam = self.cam, count = count, im = cropped, id = index, issave = self.issave)
                        """TODO
                            1. sync: 다른 카메라의 active중 가장 가까운 id를 찾는다. (synced_id)
                                적당한게 있으면 sync에 추가한다: 
                            1-1. synced_id가 reid를 가지면 이를 반환하고고
                            2. 각 id는 sync와 상관 없이 갤러리에서 reid를 시도한다
                            3. 일정 시간 안에 reid를 성공하면 sync 포함 모두 reid를 할당하고, 시간이 지나면 새로운 id를 sync에게 갱신한다.
                        """

                    
                    
                    
                    if reid in (-1,index):
                        cropped = frame[y1:y2, x1:x2]
                        if cropped is not None and cropped.size > 0:
                            self.manager.update(im = cropped, id = index, cam = self.cam, count = count, issave = self.issave)                              
                        if index in Features.fs:
                            if reid == -1 and Features.fs[index].shape[0]<10:
                                self.manager.remap_id(index)
                            if reid == -1 and Features.fs[index].shape[0]>2:
                                self.manager.sync_id(index, self.cam)
            except queue.Empty:
                LOGGER.info(f"💬 ReID queue for cam {self.cam} is empty")
        LOGGER.info(f"👋 ReID Thread    for cam {self.cam} is closed")
