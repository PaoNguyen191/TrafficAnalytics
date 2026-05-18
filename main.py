import cv2
import config
from detector.yolo_detector import Detector
from tracker.tracker import Tracker
from speed.perspective import PerspectiveTransformer
from speed.speed_estimator import SpeedEstimator
from utils.geometry import get_bottom_center
from utils.drawing import Visualizer
from utils.fps import FPSCounter
from collections import deque

def main():
    # Khởi tạo các thành phần hệ thống
    detector = Detector(config.MODEL_PATH, config.CONFIDENCE_THRESHOLD, config.TARGET_CLASSES)
    tracker = Tracker(detector)
    transformer = PerspectiveTransformer(config.SOURCE_POINTS, config.TARGET_POINTS)
    estimator = SpeedEstimator(config.PIXEL_TO_METER, config.FPS_SMOOTHING_WINDOW)
    fps_counter = FPSCounter()
    
    class_names = detector.get_names()
    trajectory_history = {} # track_id -> deque lưu lịch sử tọa độ
    
    # --- BIẾN ĐẾM XE ---
    counted_ids = set() # Lưu các ID đã đếm
    class_counts = {cls_id: 0 for cls_id in config.TARGET_CLASSES} 
    
    # --- VÙNG ĐO LƯỜNG TỐC ĐỘ (Chống nhiễu ở xa) ---
    Y_MIN_MEASURE = 1200
    Y_MAX_MEASURE = 2800

    # Thiết lập Video
    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("Đang khởi động hệ thống Phân tích Giao thông...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Nhận diện và theo dõi
        results = tracker.process_frame(frame)
        
        # 2. Vẽ đa giác hiệu chuẩn
        Visualizer.draw_homography_polygon(frame, config.SOURCE_POINTS)
        
        # 3. Vẽ vạch đếm xe
        p1, p2 = config.COUNTING_LINE
        cv2.line(frame, p1, p2, config.COLOR_LINE, 4)
        cv2.putText(frame, "Counting Line", (p1[0], p1[1] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.COLOR_LINE, 3)

        if results.boxes is not None and results.boxes.id is not None:
            # Chuyển toàn bộ dữ liệu sang NumPy
            import torch

            boxes     = torch.as_tensor(results.boxes.xyxy).cpu().numpy()
            track_ids = torch.as_tensor(results.boxes.id).int().cpu().numpy()
            class_ids = torch.as_tensor(results.boxes.cls).int().cpu().numpy()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # Tâm gầm xe
                bc_point = get_bottom_center(box)
                
                # Cập nhật quỹ đạo
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = deque(maxlen=30)
                trajectory_history[track_id].append(bc_point)
                
                # --- LOGIC ĐẾM XE ---
                if track_id not in counted_ids and len(trajectory_history[track_id]) >= 2:
                    prev_point = trajectory_history[track_id][-2]
                    curr_point = trajectory_history[track_id][-1]
                    
                    y_line = p1[1]
                    x_min, x_max = p1[0], p2[0]
                    
                    # Kiểm tra giao cắt trục Y và nằm trong giới hạn trục X
                    crossed_y = (prev_point[1] <= y_line and curr_point[1] > y_line) or \
                                (prev_point[1] >= y_line and curr_point[1] < y_line)
                    in_x_range = x_min <= curr_point[0] <= x_max
                    
                    if crossed_y and in_x_range:
                        counted_ids.add(track_id)
                        class_counts[class_id] += 1
                
                # --- LOGIC TÍNH TỐC ĐỘ (Lọc theo vùng) ---
                speed = 0.0
                if Y_MIN_MEASURE < bc_point[1] < Y_MAX_MEASURE:
                    bev_point = transformer.transform_point(bc_point)
                    speed = estimator.update(track_id, bev_point)
                
                # --- VISUALIZATION ---
                cls_name = class_names[class_id]
                Visualizer.draw_trajectory(frame, trajectory_history[track_id])
                
                # Vẽ Box (Đổi màu nếu đã đếm)
                if track_id in counted_ids:
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 150, 0), 3)
                    Visualizer.draw_bbox(frame, box, track_id, speed, cls_name)
                else:
                    Visualizer.draw_bbox(frame, box, track_id, speed, cls_name)

        # 4. Vẽ bảng thống kê số lượng phương tiện
        y_offset = 120
        cv2.putText(frame, "VEHICLES COUNT:", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        for cls_id, count in class_counts.items():
            y_offset += 60
            text = f"- {class_names[cls_id].upper()}: {count}"
            cv2.putText(frame, text, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        # 5. Cập nhật FPS
        current_fps = fps_counter.update()
        Visualizer.draw_fps(frame, current_fps)

        # 6. Lưu Frame 4K Gốc
        out.write(frame)
        
        # 7. Thu nhỏ hiển thị trên màn hình
        scale_percent = 0.33
        display_frame = cv2.resize(frame, (0, 0), fx=scale_percent, fy=scale_percent)
        cv2.imshow("Traffic Analytics", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Báo cáo tổng kết
    print("\n=== TỔNG KẾT BÁO CÁO GIAO THÔNG ===")
    for cls_id, count in class_counts.items():
        print(f"{class_names.get(cls_id, 'Unknown').capitalize()}: {count}")

if __name__ == "__main__":
    main()