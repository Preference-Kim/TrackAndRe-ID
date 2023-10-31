import sys, time

from threading import active_count
import queue 
import torch

"""import reid library"""
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID")
#from torchreid.utils import FeatureExtractor

"""import my library"""
sys.path.append("/home/sunhokim/Documents/mygit/TrackAndRe-ID/mylibrary")
from mylibrary import ReIDManager, LoadStreams, TrackCamThread, MakeVideo
from mylibrary.utils.loader_util import get_pixel_params,  get_pixel_params_filtered, get_pixel_params_mask
from mylibrary.utils import LOGGER
from mylibrary.nets import FeatureExtractor

LOGGER.info(f"\nINIT::::💡 current number of running threads: {active_count()}\n")

"""0. system params"""
is_save = True
is_record = True
min_dist_thres = 0.08
max_dist_thres = 0.14
buf_dir = f'images/gallery_{min_dist_thres}_{max_dist_thres}'

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
extractors = []
for src in rtsp_sources:
    pixel_mean, pixel_std = get_pixel_params_mask(sources=src, vid_stride=3, count=5, threshold=(235, 235, 235))
    extractors.append(
            FeatureExtractor(
                model_name='osnet_ain_x1_0',
                model_path='/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/myweights/model1031_cuda0.pth',#'/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/mars-msmt.pth.tar-60'
                verbose=False,
                num_classes=5638, # refer to dictionary in pth file
                pixel_norm=True,
                pixel_mean=pixel_mean,
                #pixel_std=pixel_std,   
                device='cuda:0' #'cuda:0'
            ))

# 2.2. YOLO
model = torch.load('weights/yolo/v8_l.pt', map_location='cuda')['model'].float()
model.eval()
model.half()

"""3. Reid manager(ReID)"""

reid_mans = []
for i,extr in enumerate(extractors):
    reid_mans.append(ReIDManager(model=extr, buf_dir=buf_dir))
    reid_mans[i].min_dist_thres = min_dist_thres 
    reid_mans[i].max_dist_thres = max_dist_thres 
LOGGER.info("")

"""4. Stream loader"""

# Create a queue for frames to be displayed in the main thread
output_queues = [queue.Queue() for _ in rtsp_sources]

# Initialize the LoadStreams class
streams = LoadStreams(sources=rtsp_sources, buffersz=15, iswait=True)
num_src = len(rtsp_sources)
fps = streams.fps[0]
resolution = streams.shape[0]

# Create VideoWriters for each source
video_man = MakeVideo(isrecord=is_record, mode='monoview', num_src=num_src, res=resolution, fps=fps, outdir='/home/sunhokim/Pictures')

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
        conf=0.03, iou=0.85 # original case: conf_threshold=0.25, iou_threshold=0.45
        )
    thread.save = is_save # whether save images used for features
    thread.stride = 8 #8
    thread.reid_man = reid_mans[i]
    thread.daemon = True
    threads.append(streams.threads[i])
    threads.append(thread)

# Start threads
for n,t in enumerate(threads):
    t.start()
    if n%2-1 == 0:
        LOGGER.info(f'Streaming and TrackCam Thread for cam {n//2} has been successfully started ✅')

LOGGER.info(f"\nSTARTING::::💡 current number of running threads: {active_count()}\n")

"""6. Displaying"""

# display and save(optional) frames
video_man.monitoring(streams=streams, frames_queue=output_queues)

time.sleep(5)

LOGGER.info(f"\nFINISHED::::💡 current number of running threads: {active_count()}")