from ultralytics import YOLO

class ChildModel:
    def __init__(self, model_path, conf=.6):
        """Initialize the YOLO-based child detector/tracker.

        Args:
            model_path: Path to a YOLO model file (e.g., .pt) compatible with
                the Ultralytics `YOLO` API.
            conf: Minimum confidence threshold for returned detections.
        """
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.conf = conf

    def child_track(self, image):
        """Track children in an image or video source and return detections.

        This method runs tracking with the BoT-SORT tracker and filters
        detections by the configured confidence threshold and the child class
        (class index 1).

        Args:
            image: Image data, a file path, or a stream/URL supported by the
                Ultralytics `YOLO.track` API.

        Returns:
            list[list]: A list of detections in the form
            ``[track_id, class_index, confidence_score, bbox_xyxy]`` where
            ``bbox_xyxy`` is ``[x1, y1, x2, y2]``.
        """
        results = self.model.track(
            image,
            tracker='botsort.yaml',
            save=False,
            conf=self.conf,
            classes=[1]
        )[0]

        boxes = results.boxes.cpu()
        detections = []
        if boxes.is_track:
            ids = boxes.id.numpy()
            cls = boxes.cls.numpy()
            confs = boxes.conf.numpy()
            xyxys = boxes.xyxy.numpy()

            for id, cl, conf, xyxy in zip(ids, cls, confs, xyxys):
                detections.append(list([int(id), int(cl), float(conf), xyxy.tolist()]))

        return detections
