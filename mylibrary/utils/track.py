from threading import Thread

import random
import cv2
import numpy as np
import torch

from . import bt_util, MyQueue, LOGGER
from .feature_manager import ReidMap, Features
from ..nets import nn as bt

class TrackCamThread(Thread):
    def __init__(self, model, streams, camid, sz, output_queue, conf=0.001, iou=0.1, isreid=True, queue_capacity=0):
        super(TrackCamThread, self).__init__()
        self.model = model
        self.cam = camid
        self.fps = streams.fps[camid]
        self.sz = sz
        self.running = streams.running
        self.input_queue = streams.queues[camid]
        self.output_queue = output_queue
        self.conf = conf
        self.iou = iou
        self.bt = bt.BYTETracker(self.fps)
        self.count = -1
        ###
        self.isreid = isreid
        self.reid_queue = MyQueue(maxsize=queue_capacity) if self.isreid else None
        self.frame_ant = None
        self.reid_stride = 3
        self.daemon = True
    
    def run(self):
        while self.running():
            self.count += 1
            frame = self.input_queue.get()
            outputs = self.track(frame)
            self.frame_ant = frame.copy()
            if len(outputs) > 0:
                boxes = outputs[:, :4]
                identities = outputs[:, 4]
                object_classes = outputs[:, 6]
                xys, indices = [], []
                for i, box in enumerate(boxes):
                    if object_classes[i] != 0:  # 0 is for person class (COCO)
                        continue
                    x1, y1, x2, y2 = list(map(int, box))
                    xys.append((x1, y1, x2, y2))
                    if identities is not None:
                        index = int(identities[i])
                        indices.append(index)
                    if index in ReidMap.id_map.keys():
                        draw_line_sync(self.frame_ant, x1, y1, x2, y2, ReidMap.id_map[index])
                    else:
                        draw_line_unsync(self.frame_ant, x1, y1, x2, y2, index)
                if self.isreid and self.reid_queue.ready and self.count%self.reid_stride == 0 :
                    msg = (frame, self.count, xys, indices)
                    self.reid_queue.put(msg)
            self.output_queue.put(self.frame_ant) # Send the frame to the main thread for displaying
        LOGGER.info(f"👋 Track Thread   for cam {self.cam} is closed")

    def track(self,frame):
        boxes = []
        confidences = []
        object_classes = []

        image = frame.copy()
        shape = image.shape[:2]

        r = self.sz / max(shape[0], shape[1])
        if r != 1:
            h, w = shape
            image = cv2.resize(image,
                            dsize=(int(w * r), int(h * r)),
                            interpolation=cv2.INTER_LINEAR)

        h, w = image.shape[:2]
        image, ratio, pad = bt_util.resize(image, self.sz)
        shapes = shape, ((h / shape[0], w / shape[1]), pad)
        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = np.ascontiguousarray(sample)
        sample = torch.unsqueeze(torch.from_numpy(sample), dim=0)

        sample = sample.cuda()
        sample = sample.half()  # uint8 to fp16/32
        sample = sample / 255  # 0 - 255 to 0.0 - 1.0

        # Inference
        with torch.no_grad():
            outputs = self.model(sample)

        # NMS
        outputs = bt_util.non_max_suppression(outputs, self.conf, self.iou) #outputs, conf_threshold=0.25, iou_threshold=0.45
        for i, output in enumerate(outputs):
            detections = output.clone()
            bt_util.scale(detections[:, :4], sample[i].shape[1:], shapes[0], shapes[1])
            detections = detections.cpu().numpy()
            for detection in detections:
                x1, y1, x2, y2 = list(map(int, detection[:4]))
                boxes.append([x1, y1, x2, y2])
                confidences.append(detection[4])
                object_classes.append(detection[5])
        outputs = self.bt.update(np.array(boxes),
                                np.array(confidences),
                                np.array(object_classes))
        return outputs
        
def draw_line_unsync(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (0, 180, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 100, 100), 2)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 2)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 2)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 2)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 2)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 2)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 2)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 2)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 2)

    text = f'ID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 2),
                0, 1 / 2, color,
                thickness=1, lineType=cv2.FILLED)

def draw_line_sync(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    random.seed(index)
    color = (random.randint(30, 255), random.randint(30, 255), random.randint(30, 255))
    color_edge = (0,250,30)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color_edge, 4)
    cv2.line(image, (x1, y1), (x1, y1 + h), color_edge, 4)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color_edge, 4)
    cv2.line(image, (x2, y1), (x2, y1 + h), color_edge, 4)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color_edge, 4)
    cv2.line(image, (x2, y2), (x2, y2 - h), color_edge, 4)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color_edge, 4)
    cv2.line(image, (x1, y2), (x1, y2 - h), color_edge, 4)

    text = f'ReID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 10),
                0, 2/3, color, #0, 1/2, color,
                thickness=3, lineType=cv2.FILLED) #thickness=1