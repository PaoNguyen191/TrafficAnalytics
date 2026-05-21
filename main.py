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

# Cache model ở phạm vi toàn cục để không phải load lại YOLO mỗi lần refresh trình duyệt
detector: Optional[Detector] = None
tracker: Optional[Tracker] = None

def get_models() -> Tuple[Detector, Tracker]:
    global detector, tracker
    if detector is None:
        detector = Detector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.TARGET_CLASSES)
        tracker = Tracker(detector)
    assert detector is not None and tracker is not None
    return detector, tracker

def generate_frames():
    # Load model từ bộ nhớ đệm
    det, trk = get_models()
    
    # Đọc cấu hình động từ File JSON (Web vừa lưu xong)
    source_points = config.SOURCE_POINTS
    counting_region = config.COUNTING_REGION
    input_video = config.INPUT_VIDEO
    
    if os.path.exists("points_config.json"):
        try:
            with open("points_config.json", "r") as f:
                data = json.load(f)
                source_points = np.array(data["SOURCE_POINTS"], dtype=np.float32)
                counting_region = np.array(data["COUNTING_REGION"], dtype=np.int32)
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
    
    Y_MIN_MEASURE = 600
    Y_MAX_MEASURE = 3500

    cap = cv2.VideoCapture(input_video)
    
    # Vẫn ghi ra file song song với luồng stream để sau này xem lại
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    fourcc = cv2.VideoWriter.fourcc(*'mp4v') 
    out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("Bắt đầu phát luồng trực tiếp (Live Stream)...")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1 

        results = trk.process_frame(frame)
        
        Visualizer.draw_homography_polygon(frame, source_points)
        cv2.polylines(frame, [counting_region], isClosed=True, color=config.COLOR_LINE, thickness=3)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [counting_region], config.COLOR_LINE)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        if results.boxes is not None and results.boxes.id is not None:
            boxes     = torch.as_tensor(results.boxes.xyxy).cpu().numpy()
            track_ids = torch.as_tensor(results.boxes.id).int().cpu().numpy()
            class_ids = torch.as_tensor(results.boxes.cls).int().cpu().numpy()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                bc_point = get_bottom_center(box)
                
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = deque(maxlen=30)
                trajectory_history[track_id].append(bc_point)
                
                if track_id not in counted_ids:
                    pt = (float(bc_point[0]), float(bc_point[1]))
                    is_inside = cv2.pointPolygonTest(counting_region, pt, False) >= 0
                    if is_inside:
                        counted_ids.add(track_id)
                        class_counts[class_id] += 1
                
                speed = 0.0
                if Y_MIN_MEASURE < bc_point[1] < Y_MAX_MEASURE:
                    bev_point = transformer.transform_point(bc_point)
                    speed = estimator.update(track_id, bev_point, frame_count, fps)
                
                cls_name = class_names[class_id]
                Visualizer.draw_trajectory(frame, trajectory_history[track_id])
                
                if track_id in counted_ids:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 150, 0), 3)
                    Visualizer.draw_bbox(frame, box, track_id, speed, cls_name)
                else:
                    Visualizer.draw_bbox(frame, box, track_id, speed, cls_name)

        y_offset = 120
        cv2.putText(frame, "VEHICLES COUNT:", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        for cls_id, count in class_counts.items():
            y_offset += 60
            text = f"- {class_names[cls_id].upper()}: {count}"
            cv2.putText(frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        current_fps = fps_counter.update()
        Visualizer.draw_fps(frame, current_fps)
        out.write(frame)

        # --- LOGIC NÉN BẢN TIN ĐỂ STREAM LÊN WEB ---
        # Thu nhỏ khung hình (Scale xuống 50%) để đường truyền Web không bị quá tải khi Stream 4K
        scale_percent = 0.5 
        stream_frame = cv2.resize(frame, (0, 0), fx=scale_percent, fy=scale_percent)

        # Encode frame ảnh thành chuẩn JPEG
        ret_encode, buffer = cv2.imencode('.jpg', stream_frame)
        if ret_encode:
            frame_bytes = buffer.tobytes()
            # Bơm luồng byte ảnh ra ngoài theo giao thức HTTP đa phần (Multipart)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()
    print("Luồng Stream kết thúc!")

if __name__ == "__main__":
    print("Vui lòng khởi chạy thông qua máy chủ FastAPI (uvicorn main_api:app) để xem luồng Stream.")