from collections import deque
import threading
import numpy as np
import cv2


class CameraViewer:

    def __init__(
        self,
        detector,
        config,
        display
    ):

        self.object_pos = dict()

        self.detector = detector
        self.display  = display
        self.callback = None
        self.running  = False
        self.thread   = None

        self.centroid_history  = deque(maxlen=8)
        self.direction_history = deque(maxlen=8)

        self.cap = cv2.VideoCapture(config.device)

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            config.width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            config.height,
        )

    def get_object(self):
        return self.object_pos

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread  = threading.Thread(
            target = self._run,
            daemon = True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

        self.cap.release()
        cv2.destroyAllWindows()

    def draw(
        self,
        image,
        result,
    ):

        if result is None:
            return image

        x1, y1, x2, y2 = result["bbox"]

        if self.display.show_roi:

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                self.display.roi_color,
                self.display.thickness,
            )

        cx, cy = result["centroid"]

        if self.display.show_centroid:

            cv2.circle(
                image,
                (cx, cy),
                5,
                self.display.centroid_color,
                -1,
            )

        if (
            self.display.show_direction
            and result["direction"] is not None
        ):

            dx, dy = result["direction"]

            end = (
                int(cx + dx * self.display.arrow_length),
                int(cy + dy * self.display.arrow_length),
            )

            cv2.arrowedLine(
                image,
                (cx, cy),
                end,
                self.display.direction_color,
                self.display.thickness,
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

    def select_object(self, boxes):

        if len(boxes) == 0:
            return None

        if self.last_centroid is None:
            box = boxes[0]
        else:
            best_dist = float("inf")
            best_box = None

            for box in boxes:
                x1, y1, x2, y2 = box

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                dist = np.hypot(
                    cx - self.last_centroid[0],
                    cy - self.last_centroid[1],
                )

                if dist < best_dist:
                    best_dist = dist
                    best_box = box

            box = best_box

        x1, y1, x2, y2 = box

        self.last_centroid = (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

        return box

    def _run(self):

        roi_size = self.display.roi_size

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                break

            result = self.detector.process(frame)
            result = self.filter_roi_size(result, roi_size)

            if result is not None:
                result["centroid"] = self.filter_centroid(result["centroid"])
                result["direction"] = self.filter_direction(result["direction"])

                self.object_pos = {
                    "centroid"  : result["centroid"],
                    "direction" : result["direction"]
                    }

            frame = self.draw(
                frame,
                result,
            )

            cv2.imshow(
                "ObjectViewer",
                frame,
            )

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
