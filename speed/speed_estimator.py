from collections import deque
import numpy as np

class SpeedEstimator:
    def __init__(self, pixel_to_meter: float, window_size: int = 15):
        self.pixel_to_meter = pixel_to_meter
        self.window_size = window_size
        self.track_history = {}
        self.speeds = {}

    def update(self, track_id: int, bev_point: tuple[float, float], frame_count: int, video_fps: float) -> float:
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.window_size)
            self.speeds[track_id] = 0.0
            
        history = self.track_history[track_id]
        history.append((frame_count, bev_point[0], bev_point[1]))

        # SỬA: Bắt đầu tính tốc độ ngay khi có đủ 3 frames (không cần đợi đủ window_size)
        if len(history) >= 3: 
            f1, x1, y1 = history[0]
            f2, x2, y2 = history[-1]
            
            # Tính khoảng thời gian giữa frame cũ nhất và mới nhất
            time_diff = (f2 - f1) / (video_fps if video_fps > 0 else 30.0)
            
            if time_diff > 0:
                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                meter_dist = pixel_dist * self.pixel_to_meter
                speed_mps = meter_dist / time_diff
                speed_kmh = speed_mps * 3.6
                
                # Loại bỏ nhiễu nếu tốc độ vọt lên quá vô lý (lớn hơn 200km/h)
                if speed_kmh < 200:
                    if self.speeds[track_id] == 0.0:
                        self.speeds[track_id] = speed_kmh
                    else:
                        # Làm mượt (Exponential Moving Average)
                        self.speeds[track_id] = (self.speeds[track_id] * 0.7) + (speed_kmh * 0.3)
                
        return self.speeds[track_id]