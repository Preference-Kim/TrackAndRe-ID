import os, datetime
import cv2
import numpy as np

from . import LOGGER

class MakeVideo:
    
    _grid_map = { # num_frame(int) : (num_rows(int), num_cols(int)) (tuple)
            1:(1,1),
            2:(1,2),
            3:(2,2),
            4:(2,2),
            5:(2,3),
            6:(2,3), # 6 is maximum for now
        }
    _imgsz = (
        (720,480),
        (854,480),
        (1280,720),
        ) # reverted for resizing function
    
    def __init__(self,isrecord=True, mode='monoview', num_src=None, res=None, fps=None, outdir='/home/sunho/Pictures'):
        assert mode in ['monoview', 'multiview'], "Initialization Error(MakeVideo): Invalid mode. Please choose 'monoview' or 'multiview'."
        self.isrecord = isrecord
        self.mode = mode
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.num_src = num_src
        self.grid_sz, self.imgsz, self.rows, self.cols = MakeVideo.get_multiview_params(num_src=num_src)
        if isrecord:
            _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            now = datetime.datetime.now()
            if mode == 'monoview':
                file_name = now.strftime("%Y-%m-%d-%H%M_record.mp4")
                output_file = os.path.join(outdir, file_name)
                self.video_writers = [cv2.VideoWriter(output_file, _fourcc, fps, self.grid_sz)]
            elif mode == 'multiview':
                base_name = now.strftime("%Y-%m-%d-%H%M_record")
                self.video_writers = []
                for n in range(num_src):
                    output_file = os.path.join(outdir, f"{base_name}_cam{n}.mp4")
                    video_writer = cv2.VideoWriter(output_file, _fourcc, fps, res[::-1])
                    self.video_writers.append(video_writer)
    
    @staticmethod
    def get_multiview_params(num_src):
        (num_rows, num_cols) = MakeVideo._grid_map[num_src]
        imgsz = MakeVideo._imgsz[1] if num_src<=4 else MakeVideo._imgsz[0]
        grid_height = imgsz[1]*num_rows
        grid_width = imgsz[0]*num_cols
        return (grid_width, grid_height), imgsz, num_rows, num_cols
    
    def monitoring(self, streams, frames_queue):
        """Create a main thread for displaying frames"""
        if self.mode == 'monoview':
            while True:
                frames = [q.get() for q in frames_queue]
                (grid_width, grid_height) = self.grid_sz
                grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

                for i, frame in enumerate(frames):
                    if frame is not None:
                        # Resize each frame to half its original size
                        resized_frame = cv2.resize(frame, self.imgsz)
                        
                        # Calculate the row and column index for the grid
                        row = i // self.cols
                        col = i % self.cols
                        
                        # Define the region to place the frame in the grid
                        y1 = row * resized_frame.shape[0]
                        y2 = (row + 1) * resized_frame.shape[0]
                        x1 = col * resized_frame.shape[1]
                        x2 = (col + 1) * resized_frame.shape[1]
                        
                        # Place the resized frame in the grid
                        grid[y1:y2, x1:x2] = resized_frame

                # Display the grid in a single window
                cv2.imshow("RTSP Streams Grid", grid.astype('uint8'))

                # Write the frame to the VideoWriter
                if self.isrecord and self.video_writers[0] is not None:
                    self.video_writers[0].write(grid)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if self.isrecord:# Release VideoWriters and close OpenCV windows
                        for video_writer in self.video_writers:
                            if video_writer is not None:
                                video_writer.release()
                        LOGGER.info(f'\n🎁 Result Video has been successfully saved \n   out_dir:{self.outdir}')

                    LOGGER.info(f"\ndeinitializing,,,")
                    streams.close()
                    cv2.destroyAllWindows()
                    break
        elif self.mode == 'multiview':
            while True:
                frames = [q.get() for q in frames_queue]

                for i, frame in enumerate(frames):
                    window_name = f"RTSP Stream {i}"
                    if frame is not None:
                        cv2.imshow(window_name, frame.astype('uint8'))
                        
                        # Write the frame to the VideoWriter
                        if self.isrecord and i < len(self.video_writers) and self.video_writers[i] is not None:
                            self.video_writers[i].write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if self.isrecord: # Release VideoWriters and close OpenCV windows
                        for video_writer in self.video_writers:
                            if video_writer is not None:
                                video_writer.release()

                    LOGGER.info(f"\ndeinitializing,,,")
                    streams.close()
                    cv2.destroyAllWindows()
                    break