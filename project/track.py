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

from threading import Thread, active_count
import queue 

import cv2
import numpy as np
import torch

from mylibrary import track_cam
from mylibrary.utils.loader import LoadStreams  # Import the LoadStreams class from the specified module
from mylibrary.utils import bt_util, LOGGER
from mylibrary.nets import nn as bt

model = torch.load('/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/yolo/v8_n.pt', map_location='cuda')['model'].float()
model.eval()
model.half()

# List of multiple RTSP sources
rtsp_sources = [
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/101',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/201',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/301',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/401',
]

# Initialize the LoadStreams class
streams = LoadStreams(sources=rtsp_sources, imgsz=640, buffer=True)

print('streams.imgsz', streams.imgsz)

# Create a queue for frames to be displayed in the main thread
frames_queue = [queue.Queue() for _ in rtsp_sources]

# Create a generator for VideoCapture objects
capture_gen = streams.cap_gen()

# Create and start threads for each video source
threads = []
for i, cap in enumerate(capture_gen):
    thread = Thread(target=track_cam, args=(model, i, streams.fps[i], streams.imgsz, cap, frames_queue, 0.2, 0.55))
    LOGGER.info(f'Video Thread ~~~~~~~~~~~~~~~~~~~~~ #{i}')
    threads.append(thread)
    thread.start()

print(f"현재 실행 중인 쓰레드 수: {active_count()}")

# Create a main thread for displaying frames
while True:
    frames = [q.get() for q in frames_queue]

    for i, frame in enumerate(frames):
        window_name = f"RTSP Stream {i}"
        cv2.imshow(window_name, frame.astype('uint8'))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        streams.close()
        cv2.destroyAllWindows()
        break

print(f"현재 실행 중인 쓰레드 수: {active_count()}")

# Wait for all threads to complete
for thread in threads:
    thread.join()
