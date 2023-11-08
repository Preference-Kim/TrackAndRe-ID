import time

from threading import active_count
import multiprocessing
import queue 
import torch

"""import my library"""
from mylibrary import ReIDentify, IDManager, LoadStreams, TrackCamThread, ReIDThread, MakeVideo
from mylibrary.process import gen_extractor
from mylibrary.utils import LOGGER
from mylibrary.nets import get_classsz

def run():
    LOGGER.info(f"\nINIT::::💡 current number of running threads: {active_count()}\n")

    """0. system params"""

    yolo_series = 'm'
    track_conf=0.01
    track_iou=0.7
    
    is_reid = True
    reid_stride_inter = 2  #8
    reid_stride_intra = 5
    queue_capacity = 0 #0:infinite
    
    is_calibrate = True
    min_dist_thres = 0.1
    max_dist_thres = 0.2

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
    num_src = len(rtsp_sources)

    """2. MODEL"""

    # 2.1. Reid

    if is_reid:
        LOGGER.info(f"Create Feature Extractor {'with' if is_calibrate else 'without'} calibration.....")
        extractors = []
        
        weight_path = '/home/sunho/Documents/mygit/TrackAndRe-ID/weights/reid/myweights/model1106.imgtri.cuda0.pth'
        num_cls=get_classsz(fpath=weight_path)
        
        multiprocessing.set_start_method("spawn") # To use CUDA with multiprocessing, you must use the 'spawn' start method
        with multiprocessing.Pool(processes=num_src) as pool:
            extractors = pool.starmap(gen_extractor, [(src, is_calibrate, weight_path, num_cls) for src in rtsp_sources])

    # 2.2. YOLO
    yolo = torch.load(f'weights/yolo/v8_{yolo_series}.pt', map_location='cuda')['model'].float()
    yolo.eval()
    yolo.half()

    """3. Reid manager(ReID)"""
    if is_reid:
        IDManager.settings(num_cam=num_src)
        ReIDentify.set_thretholds(mind=min_dist_thres, maxd=max_dist_thres)
        reid_mans = []
        for i,extr in enumerate(extractors):
            reid_mans.append(ReIDentify(model=extr, buf_dir=buf_dir))
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
    fps = streams.fps[0]
    resolution = streams.shape[0]

    # Create VideoWriters for each source
    video_man = MakeVideo(isrecord=is_record, mode='monoview', num_src=num_src, res=resolution, fps=fps, outdir=video_dir)

    """5. Threading"""

    # Create threads for each video source
    threads = []
    TrackCamThread.settings(num_cam=num_src, reid_stepsz=reid_stride_inter)
    for i in range(num_src):
        t_track = TrackCamThread(
            model=yolo, 
            streams=streams,
            camid=i, 
            sz=640, 
            output_queue=output_queues[i], 
            conf=track_conf, iou=track_iou, # original case: conf_threshold=0.25, iou_threshold=0.45
            isreid=is_reid,
            queue_capacity=queue_capacity # infinite
            )
        
        if is_reid:
            t_reid = ReIDThread(
                camid = i,
                streams=streams,
                feature_man = reid_mans[i],
                queue = t_track.reid_queue,
                stepsize = reid_stride_intra,
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

    time.sleep(3)

    while True:
        LOGGER.info(f"\nFINISHING::::💡 current number of running threads: {active_count()}")
        time.sleep(2)
        if active_count() == 1:
            break

    LOGGER.info(f"\nDONE! 👋\n")

if __name__=='__main__':
    run()