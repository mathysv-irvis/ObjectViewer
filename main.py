from objectViewer import (
    CameraViewer,
    ObjectDetector,
    CameraConfig,
    ModelConfig,
    DisplayConfig,
)

cam_cfg   = CameraConfig()
model_cfg = ModelConfig()
disp_cfg  = DisplayConfig()

detector = ObjectDetector(
    model_cfg.model_path,
    model_cfg.conf,
    model_cfg.imgsz,
)

camera = CameraViewer(detector, cam_cfg)
camera.run(disp_cfg)
