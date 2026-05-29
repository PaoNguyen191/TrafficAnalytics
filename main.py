import cv2
import config
import json
import os
from detector.yolo_detector import Detector
from tracker.tracker import Tracker
from speed.perspective import PerspectiveTransformer
from speed.speed_estimator import SpeedEstimator
from utils.geometry import get_bottom_center
from utils.drawing import Visualizer
from utils.fps import FPSCounter
from collections import deque
import torch
import numpy as np
from typing import Optional, Tuple

detector: Optional[Detector] = None
tracker: Optional[Tracker] = None

# THÊM BIẾN TOÀN CỤC LƯU DỮ LIỆU THỐNG KÊ ĐỂ GỬI LÊN WEB (REPORT)
realtime_stats = {
    "total_vehicles": 0,
    "avg_speed": 0.0,
    "fps": 0.0,
    "details": {},
    "status": "waiting" 
}

def get_models() -> Tuple[Detector, Tracker]:
    global detector, tracker
    if detector is None:
        detector = Detector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.TARGET_CLASSES)
        tracker = Tracker(detector)
    assert detector is not None and tracker is not None
    return detector, tracker

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def generate_frames():
    global realtime_stats
    det, trk = get_models()
    
    source_points = config.SOURCE_POINTS
    counting_line = config.COUNTING_LINE
    input_video = config.INPUT_VIDEO
    
    if os.path.exists("points_config.json"):
        try:
            with open("points_config.json", "r") as f:
                data = json.load(f)
                source_points = np.array(data["SOURCE_POINTS"], dtype=np.float32)
                if "COUNTING_LINE" in data:
                    counting_line = data["COUNTING_LINE"]
                input_video = data["INPUT_VIDEO"]
        except Exception as e:
            print(f"Lỗi đọc tọa độ: {e}")

    transformer = PerspectiveTransformer(source_points, config.TARGET_POINTS)
    estimator = SpeedEstimator(config.PIXEL_TO_METER, config.FPS_SMOOTHING_WINDOW)
    fps_counter = FPSCounter()
    
    class_names = det.get_names()
    trajectory_history = {} 
    counted_ids = set() 
    class_counts = {cls_id: 0 for cls_id in config.TARGET_CLASSES} 

    cap = cv2.VideoCapture(input_video)
    
    # Lấy thông số video (chỉ giữ lại FPS để tính tốc độ)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    
    # -----------------------------------------------------------------
    # ĐÃ TẮT TÍNH NĂNG GHI VIDEO RA Ổ CỨNG ĐỂ TỐI ƯU HÓA FPS (GIẢM GIẬT LAG)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    # out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (width, height))
    # -----------------------------------------------------------------

    print("Bắt đầu phát luồng trực tiếp (Live Stream)...")
    frame_count = 0
    realtime_stats["status"] = "running"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1 

        results = trk.process_frame(frame)
        
        Visualizer.draw_homography_polygon(frame, source_points)
        
        p1, p2 = counting_line[0], counting_line[1]
        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), config.COLOR_LINE, thickness=5)

        if results.boxes is not None and results.boxes.id is not None:
            boxes     = torch.as_tensor(results.boxes.xyxy).cpu().numpy()
            track_ids = torch.as_tensor(results.boxes.id).int().cpu().numpy()
            class_ids = torch.as_tensor(results.boxes.cls).int().cpu().numpy()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                bc_point = get_bottom_center(box)
                
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = deque(maxlen=30)
                trajectory_history[track_id].append(bc_point)
                
                # --- LOGIC 1: KIỂM TRA CẮT VẠCH ĐỂ ĐẾM XE ---
                if track_id not in counted_ids:
                    if len(trajectory_history[track_id]) >= 2:
                        prev_point = trajectory_history[track_id][-2] 
                        curr_point = trajectory_history[track_id][-1] 
                        
                        line_start = counting_line[0]
                        line_end = counting_line[1]
                        
                        if intersect(line_start, line_end, prev_point, curr_point):
                            counted_ids.add(track_id)
                            class_counts[class_id] += 1

                # --- LOGIC 2: ĐO TỐC ĐỘ TRONG VÙNG ROI ---
                pt = (float(bc_point[0]), float(bc_point[1]))
                is_inside_speed_zone = cv2.pointPolygonTest(source_points, pt, False) >= 0
                
                speed = 0.0
                if is_inside_speed_zone:
                    bev_point = transformer.transform_point(bc_point)
                    valid_fps = fps if fps > 0 else 30
                    speed = estimator.update(track_id, bev_point, frame_count, valid_fps)
                
                cls_name = class_names[class_id]
                Visualizer.draw_trajectory(frame, trajectory_history[track_id])
                
                if track_id in counted_ids:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 150, 0), 3)
                    Visualizer.draw_bbox(frame, box, track_id, speed, cls_name)
                else:
                    Visualizer.draw_bbox(frame, box, track_id, speed, cls_name)

        # CẬP NHẬT THỐNG KÊ REAL-TIME ĐỂ ĐẨY LÊN WEB
        valid_speeds = [s for s in estimator.speeds.values() if s > 0]
        avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0.0
        current_fps = fps_counter.update()
        
        realtime_stats["total_vehicles"] = len(counted_ids)
        realtime_stats["avg_speed"] = round(avg_speed, 1)
        realtime_stats["fps"] = round(current_fps, 1)
        realtime_stats["details"] = {class_names[k].upper(): v for k, v in class_counts.items()}

        Visualizer.draw_fps(frame, current_fps)
        
        # ĐÃ TẮT GHI FRAME VÀO Ổ CỨNG ĐỂ TRÁNH GIẬT LAG
        # out.write(frame) 

        scale_percent = 0.5 
        stream_frame = cv2.resize(frame, (0, 0), fx=scale_percent, fy=scale_percent)

        ret_encode, buffer = cv2.imencode('.jpg', stream_frame)
        if ret_encode:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    # out.release() # TẮT GIẢI PHÓNG VIDEO WRITER
    
    realtime_stats["status"] = "finished" 
    realtime_stats["fps"] = 0.0
    print("Luồng Stream kết thúc!")