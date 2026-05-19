from collections import deque
import numpy as np

class SpeedEstimator:
    def __init__(self, pixel_to_meter: float, window_size: int = 15): # Tăng window_size lên để đỡ nhiễu
        self.pixel_to_meter = pixel_to_meter
        self.window_size = window_size
        self.track_history = {}
        self.speeds = {}

    # Thêm frame_count và video_fps vào tham số
    def update(self, track_id: int, bev_point: tuple[float, float], frame_count: int, video_fps: float) -> float:
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.window_size)
            self.speeds[track_id] = 0.0
            
        history = self.track_history[track_id]
        history.append((frame_count, bev_point[0], bev_point[1]))

        # Đợi đủ số lượng frame trong window mới bắt đầu tính để mượt hơn
        if len(history) == self.window_size: 
            f1, x1, y1 = history[0]
            f2, x2, y2 = history[-1]
            
            # TÍNH THỜI GIAN CHUẨN CỦA VIDEO
            time_diff = (f2 - f1) / video_fps 
            
            if time_diff > 0:
                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                meter_dist = pixel_dist * self.pixel_to_meter
                speed_mps = meter_dist / time_diff
                speed_kmh = speed_mps * 3.6
                
                # Làm mượt (Exponential Moving Average)
                self.speeds[track_id] = (self.speeds[track_id] * 0.7) + (speed_kmh * 0.3)
                
        return self.speeds[track_id]