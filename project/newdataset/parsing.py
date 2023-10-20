import os
import shutil

source_dir = 'dataset-10-20'

# 대상 디렉토리 경로
target_dirs = [f'{source_dir}/bounding_box_train', f'{source_dir}/bounding_box_test']

# 이미지 파일 목록
image_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg')])

# 대상 디렉토리 번갈아가며 선택
current_target_dir_index = 0

# 대상 디렉토리에 이미지를 번갈아가며 옮김
for image_file in image_files:
    source_path = os.path.join(source_dir, image_file)
    target_dir = target_dirs[current_target_dir_index]
    target_path = os.path.join(target_dir, image_file)
    
    # 이미지 파일을 대상 디렉토리로 이동
    shutil.move(source_path, target_path)
    
    # 대상 디렉토리를 번갈아가며 선택
    current_target_dir_index = (current_target_dir_index + 1) % len(target_dirs)
    
print("이미지 파일을 번갈아가며 대상 디렉토리로 이동했습니다.")
