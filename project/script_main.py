import sys, os
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID")
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID/mylibrary")

from threading import active_count
import queue 

import torch

from torchreid.utils import FeatureExtractor
from mylibrary.utils.loader import LoadStreams  # Import the LoadStreams class from the specified module
from mylibrary.utils.loader_util import get_pixel_params
from mylibrary.utils import LOGGER
from mylibrary.utils.track_reid import TrackCamThread
from mylibrary.utils.gallery_manager import ReIDManager
from mylibrary.utils.display import MakeVideo

"""1. RTSP sources"""
# List of multiple RTSP sources
rtsp_sources = [
    #'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/101',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/201',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/301',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/401',
    'rtsp://admin:1234567s@10.160.30.13:554/Streaming/Channels/501',
]

"""2. MODEL"""

# 2.1. Reid
LOGGER.info("Create Feature Extractor with calibration.....")
pixel_mean, pixel_std = get_pixel_params(sources=rtsp_sources, vid_stride=1)
extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path='/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/model1024.pth.tar-600',#'/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/mars-msmt.pth.tar-60'
    pixel_mean=pixel_mean,
    pixel_std=pixel_std ,   
    device='cuda' #'cuda:0'
)

# 2.2. YOLO
model = torch.load('weights/yolo/v8_n.pt', map_location='cuda')['model'].float()
model.eval()
model.half()

"""3. Reid manager(ReID)"""

buf_dir = 'images/gallery'
reid_man = ReIDManager(model=extractor, buf_dir=buf_dir)
reid_man.min_dist_thres = 0.08
reid_man.max_dist_thres = 0.22

"""4. Stream loader and cap generator"""

# Create a queue for frames to be displayed in the main thread
output_queues = [queue.Queue() for _ in rtsp_sources]

# Initialize the LoadStreams class
streams = LoadStreams(sources=rtsp_sources, buffersz=30, iswait=True)
num_src = len(rtsp_sources)
fps = streams.fps[0]
resolution = streams.shape[0]

# Create VideoWriters for each source
video_man = MakeVideo(isrecord=False, mode='monoview', num_src=num_src, res=resolution, fps=fps, outdir='/home/sunhokim/Pictures')

"""5. Threading"""

# Create threads for each video source
threads = []
for i in range(len(streams.sources)):
    thread = TrackCamThread(
        model=model, 
        streams=streams, 
        idx=i, 
        sz=640, 
        output_queue=output_queues[i], 
        conf=0.1, iou=0.85 # original case: conf_threshold=0.25, iou_threshold=0.45
        )
    thread.save = False # whether save images used for features
    thread.stride = 30 #8
    thread.reid_man = reid_man
    thread.buf_dir = 'images/gallery'
    thread.daemon = True
    threads.append(streams.threads[i])
    threads.append(thread)

# Start threads
for n,t in enumerate(threads):
    t.start()
    if n%2-1 == 0:
        LOGGER.info(f'Streaming and TrackCam Thread for cam{n//2} has been successfully started ✅')

print(f"\nSTARTING::::💡 current number of running threads: {active_count()}\n")

"""6. Displaying"""

# display and save(optional) frames
video_man.monitoring(streams=streams, frames_queue=output_queues)

print(f"\nFINISHED::::💡 current number of running threads: {active_count()}")