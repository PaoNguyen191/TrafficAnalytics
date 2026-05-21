import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_bbox(frame: np.ndarray, bbox: list[float], track_id: int, speed: float, class_name: str, color: tuple = (0, 255, 0)):
        """Draw bounding box, ID, and speed on the frame với màu tùy chỉnh."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box bằng màu được truyền vào (thay vì cố định xanh lá)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"#{track_id} {class_name}"
        if speed > 0:
            label += f" | {speed:.1f} km/h"
            
        # Draw background and text sử dụng chung màu color cho đồng bộ
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    @staticmethod
    def draw_trajectory(frame: np.ndarray, history: list[tuple[int, int]]):
        """Draw trajectory polyline based on point history."""
        if len(history) > 1:
            points = np.array(history, dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float):
        """Draw FPS counter."""
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
    @staticmethod
    def draw_homography_polygon(frame: np.ndarray, src_points: np.ndarray):
        """Draw the calibration polygon."""
        points = src_points.astype(np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 255), thickness=2)