import numpy as np

# YOLO Configuration
MODEL_PATH = "yolo11l.pt"
CONFIDENCE_THRESHOLD = 0.4
# COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = [2, 3, 5, 7] 

# Video Configuration
INPUT_VIDEO = "videos/traffic.mp4" # Đổi lại tên file video của bạn ở đây nếu cần
OUTPUT_VIDEO = "outputs/result.mp4"

# Speed Estimation Configuration
# Hệ số: 3.5 mét / 350 pixel = 0.01 (1 pixel trong BEV = 1 cm thực tế)
PIXEL_TO_METER = 0.01 
# SỬA LẠI: Video của bạn là 25fps, nên để cửa sổ làm mượt là 25 (tương đương đúng 1 giây thực tế)
FPS_SMOOTHING_WINDOW = 25  

# Homography Points (Đã hiệu chuẩn cho video 3840x2160 Ngang)
# Lấy làn đường sát dải phân cách, chiều đi ra xa camera (ở nửa dưới màn hình)
SOURCE_POINTS = np.array([
    [2113, 1170],
    [2456, 1166],
    [3073, 2156],
    [2366, 2156],
], dtype=np.float32)

# Khung đích: Rộng 350px (tương đương bề rộng làn chuẩn 3.5m)
# Dài 2500px (Ước tính đoạn đường từ Y=1500 đến Y=2160 dài khoảng 25 mét ngoài thực tế)
TARGET_POINTS = np.array([
    [0, 0], 
    [350, 0], 
    [350, 2500], 
    [0, 2500]
], dtype=np.float32)

# SỬA LẠI VẠCH ĐẾM XE
COUNTING_REGION = np.array([
    [673, 910],
    [536, 1203],
    [2836, 1163],
    [2650, 870],
], dtype=np.int32)

# Colors
COLOR_BBOX = (0, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_TRAJECTORY = (0, 0, 255)
COLOR_LINE = (0, 255, 255) # Màu vàng