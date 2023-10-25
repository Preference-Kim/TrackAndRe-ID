"""
TODO

1. rtsp source
2. yv8:detection
3. Bytetrack
4. crop
5. re-id
"""

import sys
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID")
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID/mylibrary")

from threading import active_count
import queue 

import cv2
import numpy as np
import torch
import torch.nn as nn

import torchreid
from mylibrary.utils.loader import LoadStreams  # Import the LoadStreams class from the specified module
from mylibrary.utils import LOGGER
from mylibrary.utils.track_reid import TrackCamThread

model = torch.load('weights/yolo/v8_n.pt', map_location='cuda')['model'].float()
model.eval()
model.half()

# model_reid = torchreid.models.build_model(
#         name="osnet_ain_x1_0",
#         num_classes=1000,
#         pretrained = False
#     )
# torchreid.utils.load_pretrained_weights(model=model_reid, weight_path="weights/reid/imageNet-market.pth.tar-60")

# List of multiple RTSP sources
rtsp_sources = [
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/101',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/201',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/301',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/401',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/501',
]

# Initialize the LoadStreams class
streams = LoadStreams(sources=rtsp_sources, imgsz=640, buffer=True)

# Create a queue for frames to be displayed in the main thread
frames_queue = [queue.Queue() for _ in rtsp_sources]

# Create a generator for VideoCapture objects
capture_gen = streams.cap_gen()

# Create threads for each video source
threads = []

for i, cap in enumerate(capture_gen):
    thread = TrackCamThread(model, i, streams.fps[i], streams.imgsz, cap, frames_queue[i], 0.01, 0.85) # #outputs, conf_threshold=0.25, iou_threshold=0.45
    thread.save = True
    thread.stride = 12
    thread.buf_dir = 'images/buf-l-2023-10-20-1'
    thread.daemon = True
    threads.append(thread)

# Start threads
for t in threads:
    t.start()
    LOGGER.info(f'TrackCam Thread #{t.idx} has been successfully started ✅')

print(f"\nSTARTING::::💡 current number of running threads: {active_count()}\n")

# Create a main thread for displaying frames
while True:
    frames = [q.get() for q in frames_queue]

    for i, frame in enumerate(frames):
        window_name = f"RTSP Stream {i}"
        if frame is not None:
            cv2.imshow(window_name, frame.astype('uint8'))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        streams.close()
        cv2.destroyAllWindows()
        break

print(f"\nFINISHED::::💡 current number of running threads: {active_count()}")
