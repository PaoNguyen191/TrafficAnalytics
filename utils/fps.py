import time

class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update(self) -> float:
        """Update and return the current FPS."""
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.prev_time = current_time
        return self.fps