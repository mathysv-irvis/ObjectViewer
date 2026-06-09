
from dataclasses import dataclass


@dataclass
class CameraConfig:
    device: int = 0
    width : int = 1280
    height: int = 720


@dataclass
class ModelConfig:
    model_path: str   = "./models/runs_pipe/detect/train/weights/last.pt"
    conf      : float = 0.7
    imgsz     : int   = 640


@dataclass
class DisplayConfig:
    roi_size       : int   = 170
    show_roi       : bool  = False
    show_centroid  : bool  = True
    show_direction : bool  = True
    roi_color      : tuple = (0, 255, 0)
    centroid_color : tuple = (0, 0, 255)
    direction_color: tuple = (255, 0, 0)
    thickness      : int   = 2
    arrow_length   : int   = 100
