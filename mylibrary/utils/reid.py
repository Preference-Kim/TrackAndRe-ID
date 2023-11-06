from threading import Thread
import queue

from .feature_manager import ReidMap, Features

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
        self.manager = feature_man
        self.queue = queue
        self.issave = issave
        self.daemon = True

    def run(self):
        while self.running():
            try:
                (frame, count, boxes, indices) = self.queue.get(timeout=3)
                if len(indices) > 0:
                    self.manager.list_actives(ids=indices, cam=self.cam)
                else:
                    self.manager.list_actives(ids=[-1], cam=self.cam)
                    continue
                for index, box in zip(indices, boxes):
                    x1, y1, x2, y2 = box
                    reid = ReidMap.get_reid(index)
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
