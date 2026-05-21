import numpy as np
import json
import os

# YOLO Configuration
MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.4
TARGET_CLASSES = [2, 3, 5, 7] 

# Mặc định (như code gốc của bạn)
INPUT_VIDEO = "videos/traffic.mp4"
OUTPUT_VIDEO = "outputs/result.mp4"
PIXEL_TO_METER = 0.01 
FPS_SMOOTHING_WINDOW = 25  

# Khung đích (cố định)
TARGET_POINTS = np.array([
    [0, 0], 
    [350, 0], 
    [350, 2500], 
    [0, 2500]
], dtype=np.float32)

# Mặc định tọa độ
SOURCE_POINTS = np.array([[2113, 1170], [2456, 1166], [3073, 2156], [2366, 2156]], dtype=np.float32)
COUNTING_REGION = np.array([[673, 910], [536, 1203], [2836, 1163], [2650, 870]], dtype=np.int32)

# ==========================================
# ĐỌC CẤU HÌNH ĐỘNG TỪ GIAO DIỆN WEB (NẾU CÓ)
# ==========================================
if os.path.exists("points_config.json"):
    try:
        with open("points_config.json", "r") as f:
            config_data = json.load(f)
            SOURCE_POINTS = np.array(config_data["SOURCE_POINTS"], dtype=np.float32)
            COUNTING_REGION = np.array(config_data["COUNTING_REGION"], dtype=np.int32)
            INPUT_VIDEO = config_data["INPUT_VIDEO"]
    except Exception as e:
        print(f"Lỗi đọc file cấu hình động: {e}. Sử dụng cấu hình mặc định.")

# Colors
COLOR_BBOX = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_TRAJECTORY = (0, 0, 255)
COLOR_LINE = (0, 255, 255)