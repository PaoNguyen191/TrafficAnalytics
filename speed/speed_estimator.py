from collections import deque
from time import time
import numpy as np

class SpeedEstimator:
    def __init__(self, pixel_to_meter: float, window_size: int = 30):
        self.pixel_to_meter = pixel_to_meter
        self.window_size = window_size
        # Dictionary mapping track_id -> deque of (timestamp, bev_x, bev_y)
        self.track_history = {}
        # Dictionary mapping track_id -> smoothed speed
        self.speeds = {}

    def update(self, track_id: int, bev_point: tuple[float, float]) -> float:
        """Update track history and calculate speed."""
        current_time = time()
        
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.window_size)
            self.speeds[track_id] = 0.0
            
        history = self.track_history[track_id]
        history.append((current_time, bev_point[0], bev_point[1]))

        if len(history) >= 5: # Need minimum points for stable calculation
            # Calculate distance between oldest and newest point in the window
            t1, x1, y1 = history[0]
            t2, x2, y2 = history[-1]
            
            time_diff = t2 - t1
            if time_diff > 0:
                pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                meter_dist = pixel_dist * self.pixel_to_meter
                speed_mps = meter_dist / time_diff
                speed_kmh = speed_mps * 3.6
                
                # Simple moving average smoothing
                self.speeds[track_id] = (self.speeds[track_id] * 0.7) + (speed_kmh * 0.3)
                
        return self.speeds[track_id]