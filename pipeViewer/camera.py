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

        self.roi_size = config.roi_size

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

   def filter_roi_size(self, result):
        if result is None:
            return None

        x1, y1, x2, y2 = result["bbox"]

        width = x2 - x1
        height = y2 - y1

        if max(width, height) > self.roi_size:
            return None

        return result

    def run(
        self,
        display,
    ):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                break

            result = self.detector.process(frame)
            result = self.filter_roi_size(result)

            frame = self.draw(
                frame,
                result,
                display,
            )

            cv2.imshow(
                "PipeViewer",
                frame,
            )

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
