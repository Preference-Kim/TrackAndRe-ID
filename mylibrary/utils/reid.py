from threading import Thread

from .feature_manager import ReidMap, Features

from . import LOGGER

class ReIDThread(Thread):
    def __init__(self, streams, camid, feature_man, queue, issave):
        super(ReIDThread, self).__init__()
        self.cam = camid
        self.running = streams.running
        self.manager = feature_man
        self.queue = queue
        self.issave = issave
        self.daemon = True

    def run(self):
        while self.running():
            (frame, count, boxes, indices) = self.queue.get()
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
                    if reid == -1:                                
                        if index in Features.fs:
                            if Features.fs[index].shape[0]<10:
                                self.manager.remap_id(index)
                            if Features.fs[index].shape[0]>2:
                                self.manager.sync_id(index, self.cam)
        LOGGER.info(f"👋 ReID Thread    for cam {self.cam} is closed")
