import sys, os
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID")
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID/mylibrary")

from threading import active_count
import queue 

import cv2
import numpy as np
import torch

from torchreid.utils import FeatureExtractor
from mylibrary.utils.loader import LoadStreams  # Import the LoadStreams class from the specified module
from mylibrary.utils.loader_util import get_pixel_params
from mylibrary.utils import LOGGER
from mylibrary.utils.track_reid import TrackCamThread
from mylibrary.utils.gallery_manager import ReIDManager

"""1. RTSP sources"""
# List of multiple RTSP sources
rtsp_sources = [
    #'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/101',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/201',
    #'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/301',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/401',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/501',
]

"""2. MODEL"""

#2.1. Reid
LOGGER.info("Create Feature Extractor with calibration.....")
pixel_mean, pixel_std = get_pixel_params(sources=rtsp_sources, vid_stride=1)
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path='/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/model1024.pth.tar-600',#'/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/mars-msmt.pth.tar-60'
    pixel_mean=pixel_mean,
    pixel_std=pixel_std ,   
    device='cuda' #'cuda:0'
)

#2.2. YOLO
model = torch.load('weights/yolo/v8_s.pt', map_location='cuda')['model'].float()
model.eval()
model.half()

"""3. Reid manager(ReID)"""

buf_dir = 'images/gallery'
reid_man = ReIDManager(model=extractor, buf_dir=buf_dir)
reid_man.min_dist_thres = 0.08
reid_man.max_dist_thres = 0.22

"""4. Stream loader and cap generator"""

# Create a queue for frames to be displayed in the main thread
frames_queue = [queue.Queue() for _ in rtsp_sources]

# Initialize the LoadStreams class
streams = LoadStreams(sources=rtsp_sources, imgsz=640, buffer=True)

# Create a generator for VideoCapture objects
capture_gen = streams.cap_gen()

# Create VideoWriters for each source
vid_record=False
if vid_record:
    video_writers = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

"""5. Threading"""

# Create threads for each video source
threads = [] 
for i, cap in enumerate(capture_gen):
    thread = TrackCamThread(model, i, streams.fps[i], streams.imgsz, cap, frames_queue[i], 0.1, 0.85) # #outputs, conf_threshold=0.25, iou_threshold=0.45
    thread.record = False 
    thread.stride = 8
    thread.reid_man = reid_man
    thread.buf_dir = 'images/gallery'
    thread.daemon = True
    threads.append(thread)
    
    if vid_record:
        # Create a VideoWriter for each source
        outdir = '/home/sunhokim/Pictures'
        output_file = os.path.join(outdir, f'2023-10-25-cam{i}.mp4')
        video_writer = cv2.VideoWriter(output_file, fourcc, thread.fps, (int(cap.get(3)),int(cap.get(4))))
        video_writers.append(video_writer)

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
            
            # Write the frame to the VideoWriter
            if vid_record and i < len(video_writers) and video_writers[i] is not None:
                video_writers[i].write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if vid_record:# Release VideoWriters and close OpenCV windows
            for video_writer in video_writers:
                if video_writer is not None:
                    video_writer.release()

        streams.close()
        cv2.destroyAllWindows()
        break

print(f"\nFINISHED::::💡 current number of running threads: {active_count()}")