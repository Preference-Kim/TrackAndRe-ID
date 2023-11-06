import os
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

from . import LOGGER

def get_pixel_params(sources, vid_stride=3, count=1):
    sources = sources if isinstance(sources, list) else Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
    
    pixel_values = []
    
    for s in sources:
        # RTSP 스트림 열기
        s = eval(s) if s.isnumeric() else s
        cap = cv2.VideoCapture(s)

        n = 0

        while n<count:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 프레임에서 픽셀 값을 가져와 리스트에 추가
            if n % vid_stride == 0:
                pixel_values.append(frame.reshape(-1, 3))
            n += 1

        # RTSP 스트림 닫기
        cap.release()
    
    # 모든 프레임에 대한 픽셀 값을 하나의 배열로 변환
    pixel_values = np.vstack(pixel_values)

    # 픽셀 평균 및 표준 편차 계산
    pixel_mean = np.mean(pixel_values, axis=0)/255.0
    pixel_std = np.std(pixel_values, axis=0)/255.0

    LOGGER.info("")
    LOGGER.info(f"📷 Pixel Mean:                {pixel_mean}")
    LOGGER.info(f"📷 Pixel Standard Deviation:  {pixel_std}\n")
    
    return pixel_mean, pixel_std

def get_pixel_params_filtered(sources, vid_stride=3, count=1, z_score_threshold=2.0):
    sources = sources if isinstance(sources, list) else Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]
    
    pixel_values = []
    
    for s in sources:
        # RTSP 스트림 열기
        s = eval(s) if s.isnumeric() else s
        cap = cv2.VideoCapture(s)

        n = 0

        while n<count:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 프레임에서 픽셀 값을 가져와 리스트에 추가
            if n % vid_stride == 0:
                pixel_values.append(frame.reshape(-1, 3))
            n += 1

        # RTSP 스트림 닫기
        cap.release()
    
    # 모든 프레임에 대한 픽셀 값을 하나의 배열로 변환
    pixel_values = np.vstack(pixel_values)

    # Z-점수를 사용하여 이상치 식별
    z_scores = stats.zscore(pixel_values)

    # 이상치가 아닌 픽셀 값만 선택
    valid_pixel_values = pixel_values[np.all(np.abs(z_scores) <= z_score_threshold, axis=1)]

    # 픽셀 평균 및 표준 편차 계산
    pixel_mean = np.mean(valid_pixel_values, axis=0)/255.0
    pixel_std = np.std(valid_pixel_values, axis=0)/255.0

    LOGGER.info("")
    LOGGER.info(f"📷 Pixel Mean (Outliers Removed):                {pixel_mean}")
    LOGGER.info(f"📷 Pixel Standard Deviation (Outliers Removed):  {pixel_std}\n")
    
    return pixel_mean, pixel_std



def get_pixel_params_mask(sources, vid_stride=3, count=1, threshold=(210, 210, 210)):
    # Convert sources to a list if it's not already
    sources = sources if isinstance(sources, list) else Path(sources).read_text().rsplit() if os.path.isfile(sources) else [sources]

    pixel_means = []
    pixel_stds = []

    for s in sources:
        # Open RTSP streams
        s = eval(s) if s.isnumeric() else s
        cap = cv2.VideoCapture(s)

        n = 0

        while n < count:
            ret, frame = cap.read()

            if not ret:
                break

            if n % vid_stride == 0:
                # Create a mask for the current frame
                mask = cv2.inRange(frame, (0,0,0), threshold)
                # Get pixel values from the frame
                pixel_values = frame[np.where(mask > 0)].reshape(-1, 3)

                # Apply the mask to the pixel values
                active_pixels = pixel_values

                if len(active_pixels) > 0:
                    # Calculate mean and standard deviation using active pixels only
                    pixel_mean = np.mean(active_pixels, axis=0) / 255.0
                    pixel_std = np.std(active_pixels, axis=0) / 255.0
                else:
                    # Handle the case when there are no active pixels
                    pixel_mean = np.array([0, 0, 0])
                    pixel_std = np.array([0, 0, 0])

                pixel_means.append(pixel_mean)
                pixel_stds.append(pixel_std)

            n += 1

        # Close RTSP streams
        cap.release()

    # Calculate the overall mean and standard deviation
    overall_mean = np.mean(pixel_means, axis=0)
    overall_mean = np.around(overall_mean, 3).tolist()
    overall_std = np.mean(pixel_stds, axis=0)
    overall_std = np.around(overall_std, 3).tolist()

    LOGGER.info("")
    LOGGER.info(f"📷 Pixel Mean (Excluding Bright Areas):                {overall_mean}")
    LOGGER.info(f"📷 Pixel Standard Deviation (Excluding Bright Areas):  {overall_std}\n")

    return (overall_mean, overall_std)