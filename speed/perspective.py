import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray):
        """Initialize the homography matrix for Bird-Eye-View transformation."""
        self.matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def transform_point(self, point: tuple[int, int]) -> tuple[float, float]:
        """Apply homography matrix to a single point."""
        point_np = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_np, self.matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])