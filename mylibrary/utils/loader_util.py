import os
from pathlib import Path

import cv2
import numpy as np

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

    LOGGER.info(f"📷 Pixel Mean:                {pixel_mean}")
    LOGGER.info(f"📷 Pixel Standard Deviation:  {pixel_std}\n")
    
    return pixel_mean, pixel_std