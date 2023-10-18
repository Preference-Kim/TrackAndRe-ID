import os
from copy import deepcopy
from threading import Thread

import cv2
import numpy as np
import torch

from . import bt_util
from ..nets import nn as bt

class TrackCamThread(Thread):
    def __init__(self, model, idx, fps, sz, cap, frames_queue, conf=0.001, iou=0.1):
        super(TrackCamThread, self).__init__()
        self.model = model
        self.idx = idx
        self.fps = fps
        self.sz = sz
        self.cap = cap
        self.frames_queue = frames_queue
        self.conf = conf
        self.iou = iou
        self.bt = bt.BYTETracker(self.fps)
        self.count = -1
        ###
        self.frame_ant = None
        self.record = True
        self.stride = 3
        self.buf_dir = 'images/buf'
    
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            else:
                self.count += 1
                outputs = self.track(frame)
                self.frame_ant = frame.copy()
                if len(outputs) > 0:
                    boxes = outputs[:, :4]
                    identities = outputs[:, 4]
                    object_classes = outputs[:, 6]
                    for i, box in enumerate(boxes):
                        if object_classes[i] != 0:  # 0 is for person class (COCO)
                            continue
                        x1, y1, x2, y2 = list(map(int, box))
                        # get ID of object
                        index = int(identities[i]) if identities is not None else 0
                        if self.count%self.stride == 0 and self.record:
                            crop_dir = f'{self.buf_dir}/{self.idx}'
                            os.makedirs(crop_dir, exist_ok=True)
                            cropped = deepcopy(frame[y1:y2, x1:x2])
                            if cropped is not None and cropped.size > 0:
                                cv2.imwrite(f'{crop_dir}/id{index}_{self.count}.jpg', cropped)
                        draw_line(self.frame_ant, x1, y1, x2, y2, index)
            
            self.frames_queue.put(self.frame_ant) # Send the frame to the main thread for displaying
    
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
        
def draw_line(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    color = (200, 0, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f'ID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 2),
                0, 1 / 2, (0, 255, 0),
                thickness=1, lineType=cv2.FILLED)