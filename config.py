import numpy as np

# YOLO Configuration
MODEL_PATH = "yolo11m.pt"
CONFIDENCE_THRESHOLD = 0.4
# COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [2, 3, 5, 7] 

# Video Configuration
INPUT_VIDEO = "videos/traffic.mp4"
OUTPUT_VIDEO = "outputs/result.mp4"

# Speed Estimation Configuration
# Hệ số: 3.5 mét / 350 pixel = 0.01 (1 pixel trong BEV = 1 cm thực tế)
PIXEL_TO_METER = 0.01 
FPS_SMOOTHING_WINDOW = 30  # Số khung hình dùng để làm mượt tốc độ trung bình

# Homography Points (Đã hiệu chuẩn cho video 4K dọc - Làn đường ở giữa)
SOURCE_POINTS = np.array([
    [980, 1800],   # Trên - Trái
    [1220, 1800],  # Trên - Phải
    [1420, 2600],  # Dưới - Phải
    [900, 2600]    # Dưới - Trái
], dtype=np.float32)

# Khung đích: Rộng 350px (tương đương 3.5m), Dài 1500px (tương đương 15m)
TARGET_POINTS = np.array([
    [0, 0], 
    [350, 0], 
    [350, 1500], 
    [0, 1500]
], dtype=np.float32)

COUNTING_LINE = [(0, 2000), (2600, 2000)]
# Colors
COLOR_BBOX = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_TRAJECTORY = (0, 0, 255)
COLOR_LINE = (0, 255, 255) # Màu vàng