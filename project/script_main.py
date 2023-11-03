import time

from threading import active_count
import queue 
import torch

"""import my library"""
from mylibrary import ReIDManager, LoadStreams, TrackCamThread, ReIDThread, MakeVideo
from mylibrary.utils.loader_util import get_pixel_params,  get_pixel_params_filtered, get_pixel_params_mask
from mylibrary.utils import LOGGER
from mylibrary.nets import FeatureExtractor, get_classsz

def run():
    LOGGER.info(f"\nINIT::::💡 current number of running threads: {active_count()}\n")

    """0. system params"""

    yolo_series = 'l'
    track_conf=0.01
    track_iou=0.7
    
    is_reid = True
    reid_stride = 8  #8
    stepsize = 4
    queue_capacity = 0 #0:infinite
    is_calibrate = False
    min_dist_thres = 0.1
    max_dist_thres = 0.15

    is_save = False
    is_record = False
    buf_dir = f'images/1103/gallery_{min_dist_thres}_{max_dist_thres}'
    video_dir = '/home/sunho/Pictures/1103'

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

    if is_reid:
        LOGGER.info("Create Feature Extractor with calibration.....")
        extractors = []
        
        weight_path = '/home/sunho/Documents/mygit/TrackAndRe-ID/weights/reid/myweights/model1103.vidsoft_cuda0.pth'
        num_cls=get_classsz(fpath=weight_path)
        
        for src in rtsp_sources:
            pixel_mean, pixel_std = get_pixel_params_mask(sources=src, vid_stride=3, count=5, threshold=(235, 235, 235)) if is_calibrate else [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # ImageNet statistics
            extractors.append(
                    FeatureExtractor(
                        model_name='osnet_ain_x1_0',
                        model_path=weight_path,#'/home/sunhokim/Documents/mygit/TrackAndRe-ID/weights/reid/mars-msmt.pth.tar-60'
                        verbose=False,
                        num_classes=num_cls, # refer to dictionary in pth file
                        pixel_norm=True,
                        pixel_mean=pixel_mean,
                        pixel_std=pixel_std,   
                        device='cuda:0' #'cuda:0'
                    ))

    # 2.2. YOLO
    yolo = torch.load(f'weights/yolo/v8_{yolo_series}.pt', map_location='cuda')['model'].float()
    yolo.eval()
    yolo.half()

    """3. Reid manager(ReID)"""
    if is_reid:
        reid_mans = []
        for i,extr in enumerate(extractors):
            reid_mans.append(ReIDManager(model=extr, buf_dir=buf_dir))
            reid_mans[i].min_dist_thres = min_dist_thres 
            reid_mans[i].max_dist_thres = max_dist_thres 
        LOGGER.info("")
        num_set=3
    else:
        LOGGER.info("💤 ReID module is deactivated. Change \'is_reid\' if you want\n")
        num_set=2

    """4. Stream loader"""

    # Create a queue for frames to be displayed in the main thread
    output_queues = [queue.Queue() for _ in rtsp_sources]

    # Initialize the LoadStreams class
    streams = LoadStreams(
        sources=rtsp_sources, 
        vid_stride=1, 
        buffersz=queue_capacity, 
        iswait=True, 
        is_stack=True
        )
    num_src = len(rtsp_sources)
    fps = streams.fps[0]
    resolution = streams.shape[0]

    # Create VideoWriters for each source
    video_man = MakeVideo(isrecord=is_record, mode='monoview', num_src=num_src, res=resolution, fps=fps, outdir=video_dir)

    """5. Threading"""

    # Create threads for each video source
    threads = []
    for i in range(len(streams.sources)):
        t_track = TrackCamThread(
            model=yolo, 
            streams=streams, 
            camid=i, 
            sz=640, 
            output_queue=output_queues[i], 
            conf=track_conf, iou=track_iou, # original case: conf_threshold=0.25, iou_threshold=0.45
            isreid=is_reid,
            reid_stride=reid_stride,
            queue_capacity=queue_capacity # infinite
            )
        
        if is_reid:
            t_reid = ReIDThread(
                camid = i,
                streams=streams,
                feature_man = reid_mans[i],
                queue = t_track.reid_queue,
                stepsize = stepsize,
                issave = is_save            # whether save images used for features
            )
        
        threads.append(streams.threads[i])
        threads.append(t_track)
        if is_reid:
            threads.append(t_reid)
        
    # Start threads
    for n,t in enumerate(threads):
        t.start()
        if n%num_set-1 == 0:
            LOGGER.info(f'Streaming and Track-ReID Thread for cam {n//num_set} has been successfully started ✅')

    LOGGER.info(f"\nSTARTING::::💡 current number of running threads: {active_count()}\n")

    """6. Displaying"""

    # display and save(optional) frames
    video_man.monitoring(streams=streams, frames_queue=output_queues)

    time.sleep(5)

    while True:
        LOGGER.info(f"\nFINISHING::::💡 current number of running threads: {active_count()}")
        time.sleep(2)
        if active_count() == 1:
            break

    LOGGER.info(f"\nDONE! 👋\n\n\n")

if __name__=='__main__':
    run()