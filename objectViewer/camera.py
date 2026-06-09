from collections import deque
import numpy as np
import cv2


class CameraViewer:

    def __init__(
        self,
        detector,
        config,
    ):

        self.detector = detector

        self.cap = cv2.VideoCapture(
            config.device
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.height,
        )

        self.centroid_history  = deque(maxlen=8)
        self.direction_history = deque(maxlen=8)

    def draw(
        self,
        image,
        result,
        display,
    ):

        if result is None:
            return image

        x1, y1, x2, y2 = result["bbox"]

        if display.show_roi:

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                display.roi_color,
                display.thickness,
            )

        cx, cy = result["centroid"]

        if display.show_centroid:

            cv2.circle(
                image,
                (cx, cy),
                5,
                display.centroid_color,
                -1,
            )

        if (
            display.show_direction
            and result["direction"] is not None
        ):

            dx, dy = result["direction"]

            end = (
                int(cx + dx * display.arrow_length),
                int(cy + dy * display.arrow_length),
            )

            cv2.arrowedLine(
                image,
                (cx, cy),
                end,
                display.direction_color,
                display.thickness,
            )

        return image

    def filter_direction(self, direction):

        if direction is None:
            return None

        self.direction_history.append(direction)

        dx = np.mean([d[0] for d in self.direction_history])
        dy = np.mean([d[1] for d in self.direction_history])

        norm = np.hypot(dx, dy)

        if norm > 0:
            dx /= norm
            dy /= norm

        return (dx, dy)

    def filter_centroid(self, centroid):

        if centroid is None:
            return None

        self.centroid_history.append(centroid)

        x = int(np.mean([c[0] for c in self.centroid_history]))
        y = int(np.mean([c[1] for c in self.centroid_history]))

        return (x, y)

    def filter_roi_size(self, result, roi_size):
        if result is None:
            return None

        x1, y1, x2, y2 = result["bbox"]

        width = x2 - x1
        height = y2 - y1

        if max(width, height) > roi_size:
            return None

        return result

    def run(
        self,
        display,
    ):
        roi_size = display.roi_size

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            result = self.detector.process(frame)
            result = self.filter_roi_size(result, roi_size)

            if result is not None:
                result["centroid"] = self.filter_centroid(result["centroid"])
                result["direction"] = self.filter_direction(result["direction"])

            frame = self.draw(
                frame,
                result,
                display,
            )

            cv2.imshow(
                "ObjectViewer",
                frame,
            )

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
