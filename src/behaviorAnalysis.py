from ultralytics import YOLO

class BehaviorAnalysisModel:
    def __init__(self, model_path, conf=.6):
        """Wrap an Ultralytics YOLO model for behavior detection.

        Args:
            model_path (str | os.PathLike[str]):
                Path to a YOLO model file (e.g., ``.pt``) compatible with the
                Ultralytics :class:`ultralytics.YOLO` API.
            conf (float):
                Minimum confidence threshold applied during prediction.
        """
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.conf = conf

    def detect_behavior(self, image):
        """Run YOLO inference and return detections.

        Args:
            image (str | os.PathLike[str] | numpy.ndarray | PIL.Image.Image):
                Image input accepted by Ultralytics YOLO (e.g., a NumPy array in
                HWC/BGR or HWC/RGB format, a file path, or a PIL Image).

        Returns:
            list[list[object]]:
                A list of detections of the form ``[class_id, confidence, [x1, y1, x2, y2]]``.
        """
        results = self.model.predict(
            image,
            save=False,
            conf=self.conf
        )[0]

        boxes = results.boxes.cpu()
        detections = []
        cls = boxes.cls.numpy()
        confs = boxes.conf.numpy()
        xyxys = boxes.xyxy.numpy()

        for cl, conf, xyxy in zip(cls, confs, xyxys):
            detections.append(list([int(cl), float(conf), xyxy.tolist()]))

        return detections
