def get_bottom_center(bbox: list[float]) -> tuple[int, int]:
    """Calculate the bottom-center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)