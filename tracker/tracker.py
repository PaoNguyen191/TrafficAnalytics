from detector.yolo_detector import Detector
import numpy as np

class Tracker:
    def __init__(self, detector: Detector):
        """Wrapper for YOLO's integrated ByteTrack logic."""
        self.detector = detector

    def process_frame(self, frame: np.ndarray):
        """Run detection and ByteTrack."""
        # Using persist=True keeps track history active between frames
        results = self.detector.model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            conf=self.detector.conf, 
            classes=self.detector.classes,
            device=self.detector.device,
            half=(self.detector.device == 'cuda'), # Use FP16 if on GPU
            imgsz=640,
            verbose=False
        )
        return results[0]