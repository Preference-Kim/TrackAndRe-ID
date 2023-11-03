from threading import Thread

from .feature_manager import ReidMap, Features

from . import LOGGER

class ReIDThread(Thread):
    def __init__(self, streams, camid, feature_man, queue, stepsize, issave):
        super(ReIDThread, self).__init__()
        self.cam = camid
        self.running = streams.running
        self.manager = feature_man
        self.queue = queue
        self.issave = issave
        self.daemon = True
        self.stepsize = stepsize

    def run(self):
        _f = Features()
        while self.running():
            (frame, count, boxes, indices) = self.queue.get()
            if len(indices) > 0:
                self.manager.list_actives(ids=indices, cam=self.cam)
                modc = count % self.stepsize
                for index, box in zip(indices, boxes):
                    reid = ReidMap.get_reid(index)
                    if reid != -1:
                        modi = index % self.stepsize
                        if modc != modi:
                            continue 
                    x1, y1, x2, y2 = box
                    cropped = frame[y1:y2, x1:x2]
                    if cropped is not None and cropped.size > 0:
                        self.manager.update(im = cropped, id = index, reid = reid, cam = self.cam, count = count, issave = self.issave)
                        print(f"count:::{count}")
                    if index in _f.fs:
                        print(f"FTS index::   {index}\nFTS[1]:: {_f(index)[1]}")
                        len_f = len(_f(index)[1])
                        if reid == -1 and len_f<5:
                            self.manager.remap_id(index)
                        if reid == -1 and len_f>3:
                            self.manager.sync_id(index, self.cam)
            else:
                self.manager.list_actives(ids=[-1], cam=self.cam)
                continue

        LOGGER.info(f"👋 ReID Thread    for cam {self.cam} is closed")
