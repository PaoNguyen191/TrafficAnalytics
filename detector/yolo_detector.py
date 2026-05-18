from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_path: str, conf_thresh: float, classes: list[int]):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        self.conf = conf_thresh
        self.classes = classes

    def get_names(self) -> dict:
        return self.model.names