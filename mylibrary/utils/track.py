from threading import Thread

import random
import cv2
import numpy as np
import torch

from . import bt_util, MyQueue, LOGGER
from .feature_manager import ReIDentify, ReidMap, IDManager, Features
from ..nets import nn as bt

class TrackCamThread(Thread):

    num_cam = 0
    stepsz = 1
    step = 1
    which_cam = -15 # reid 진행할 camid 지칭, 0 되기 전까지는 track만 진행함

    @staticmethod
    def settings(num_cam = 0, reid_stepsz = 1):
        TrackCamThread.num_cam = num_cam
        TrackCamThread.stepsz = reid_stepsz

    def __init__(
        self, model, streams, camid, sz, output_queue, conf=0.001, iou=0.1, isreid=True, queue_capacity=0, life=5):
        """
        Args:

        """
        super().__init__(daemon = True)
        self.model = model
        self.cam = camid
        self.input_queue = streams.queues[camid]
        self.running = streams.running
        self.fps = streams.fps[camid]
        """for tracking"""
        self.bt = bt.BYTETracker(self.fps)
        self.sz = sz
        self.conf = conf
        self.iou = iou
        self.count = -1
        """for reid"""
        self.reid_queue = MyQueue(maxsize=queue_capacity)
        self.frame_ant = None
        self.isreid = isreid
        """for id managing"""
        self.activeids = []
        self.deactiveids = {}      # id: [count, cam]
        self.life = life
        """output"""
        self.query = IDManager.task_q
        self.output_queue = output_queue

    @classmethod
    def ismyturn(cls, camid):
        who = cls.which_cam
        if cls.step == cls.stepsz:
            cls.step = 1
            if who in range(cls.num_cam):
                cls.which_cam = (who + 1) % cls.num_cam
                return camid==who
            elif who < 0:
                cls.which_cam += 1
            else:
                cls.which_cam = -1
        elif cls.step < cls.stepsz:
            cls.step += 1
        else:
            cls.step = 1
        return False
    
    def run(self):
        while self.running():
            frame = self.input_queue.get()
            outputs, removed_ids = self.track(frame)
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
                if self.isreid:
                    self.query.put_query(task='updateIDs', items={'cam':self.cam, 'fps':self.fps, 'active_ids':indices, 'removed_ids':removed_ids})
                    #TODO: reid는 매번 reidmap을 통해서 갱신하기(reid core, sync 분리)
                    if self.ismyturn(self.cam) and self.reid_queue.ready:
                        self.count += 1
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
        outputs, removed_ids = self.bt.update(boxes=np.array(boxes),
                                scores=np.array(confidences),
                                object_classes=np.array(object_classes),
                                get_removed_tracks=True)
        return (outputs, removed_ids)

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