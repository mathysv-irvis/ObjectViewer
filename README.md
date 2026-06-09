
# ObjectViewer

## Overview

ObjectViewer is a lightweight Python library for detecting and visualizing pipes using YOLO segmentation models and OpenCV.

The library provides a simple interface for:

* Camera acquisition
* YOLO inference
* Object detection
* Centroid computation
* Object orientation estimation
* Real-time visualization of detection results

The project is designed to be easily integrated into robotics, automation, and computer vision pipelines.

---

# Features

* YOLO segmentation inference
* Binary mask extraction
* Centroid computation using image moments
* Object direction estimation
* ROI visualization
* Centroid visualization
* Direction vector visualization
* Configurable camera and model parameters

---

# Project Structure

```text
pipeViewer/
│
├── __init__.py
├── camera.py
├── detection.py
├── config.py
└── utils.py
```

| File           | Description                                     |
| -------------- | ----------------------------------------------- |
| `camera.py`    | Camera acquisition and visualization utilities  |
| `detection.py` | YOLO inference and pipe processing algorithms   |
| `config.py`    | Camera, model and display configuration classes |
| `utils.py`     | Optional helper functions                       |

---

# Installation

Clone the repository:

```bash
git clone <repository_url>
cd pipeViewer
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

or install manually:

```bash
pip install ultralytics opencv-python numpy
```

For development, install the package in editable mode:

```bash
pip install -e .
```

---

# Basic Usage

```python
from pipeViewer import (
    Camera,
    ObjectDetector,
    CameraConfig,
    ModelConfig,
    DisplayConfig,
)

camera_cfg = CameraConfig()

model_cfg = ModelConfig(
    model_path="weights/best.pt",
    conf=0.4,
)

display_cfg = DisplayConfig()

detector = ObjectDetector(
    model_path=model_cfg.model_path,
    conf=model_cfg.conf,
    imgsz=model_cfg.imgsz,
)

camera = Camera(
    detector,
    camera_cfg,
)

camera.run(display_cfg)
```

---

# Detection Output

Each inference returns a dictionary containing the processed information:

```python
{
    "mask": mask,
    "centroid": (cx, cy),
    "direction": (dx, dy)
}
```

| Field       | Description                                   |
| ----------- | --------------------------------------------- |
| `mask`      | Binary segmentation mask                      |
| `centroid`  | Center of the detected pipe                   |
| `direction` | Unit vector representing the pipe orientation |

---

# Configuration

The library exposes dataclass-based configuration objects.

Example:

```python
camera_cfg = CameraConfig(
    device=0,
    width=1280,
    height=720,
)

model_cfg = ModelConfig(
    model_path="weights/best.pt",
    conf=0.5,
    imgsz=640,
)
```

This allows camera parameters, model settings and visualization options to be modified independently.

---

# Dependencies

* Python 3.10 or later
* NumPy
* OpenCV
* Ultralytics YOLO

---

# Planned Extensions

The current implementation serves as a foundation for additional features, including:

* Skeleton-based orientation estimation
* Multi-pipe detection and tracking
* 3D pipe pose estimation
* ROS2 integration
* Intel RealSense support
* Stereo vision support

---

# License

This project is distributed under the MIT License.
