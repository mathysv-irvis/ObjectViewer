import cv2
import numpy as np

from ultralytics import YOLO

class ObjectDetector:

    def __init__(
        self,
        model_path,
        conf=0.4,
        imgsz=640,
    ):

        self.model = YOLO(model_path)

        self.conf = conf
        self.imgsz = imgsz

    def process(self, image):

        result = self.model.track(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            persist=True,
        )[0]

        if len(result.boxes) == 0:
            return None

        box = result.boxes.xyxy.cpu().numpy()[0]

        x1, y1, x2, y2 = box.astype(int)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=20,
        )

        direction = None

        if lines is not None:

            best = None
            best_len = 0

            for l in lines:

                x3, y3, x4, y4 = l[0]

                length = np.hypot(
                    x4 - x3,
                    y4 - y3,
                )

                if length > best_len:
                    best_len = length
                    best = (x3, y3, x4, y4)

            x3, y3, x4, y4 = best

            dx = x4 - x3
            dy = y4 - y3

            norm = np.hypot(dx, dy)

            if norm > 0:
                direction = (
                    dx / norm,
                    dy / norm,
                )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        return {
            "bbox": (x1, y1, x2, y2),
            "centroid": (cx, cy),
            "direction": direction,
        }
