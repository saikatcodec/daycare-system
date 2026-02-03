from ultralytics import YOLO
import pickle
import numpy as np

class FallDetect:
    """Fall detection using YOLO pose keypoints + an SVM classifier.

    The pipeline is:
    1) Run pose inference to get keypoints + bounding boxes.
    2) Build a small, hand-crafted feature vector from keypoints/box.
    3) Standardize features with a saved scaler.
    4) Classify with a saved SVM to predict fall vs not-fall.
    """
    def __init__(self, svm_path):
        """Initialize the detector.

        Args:
            svm_path: Path to a pickled scikit-learn SVM/SVC model.
        """
        self.model = YOLO('yolo11m-pose.pt')
        self.scaler = pickle.load(open('./models/fallDetectScaler.pkl', 'rb'))
        self.svm = pickle.load(open(svm_path, 'rb'))

    def __nose_ref_point(self, point, nose):
        """Convert a keypoint to a coordinate system relative to the nose.

        Args:
            point: A 2D point (x, y) in image coordinates.
            nose: The nose keypoint (x, y) used as the origin.

        Returns:
            tuple[float, float]: The translated point (x - nose_x, y - nose_y).
        """
        new_point = (float(point[0] - nose[0]), float(point[1] - nose[1]))
        return tuple(new_point)
    
    def __get_points(self, image_path, conf=0.8):
        """Extract pose keypoints and bounding boxes from an image.

        Args:
            image_path: Path to an image (or a source supported by
                Ultralytics' `YOLO.predict`).
            conf: Confidence threshold forwarded to the YOLO predictor.

        Returns:
            tuple[np.ndarray, np.ndarray | list]:
            - keypoints: Array of detected person keypoints.
            - bboxes: Array of bounding boxes in xyxy format (int), or an empty
              list if no detections.
        """
        results = self.model.predict(image_path, conf=conf, imgsz=640, verbose=False)
        keypoints = results[0].keypoints.xy.cpu().numpy()
        if len(keypoints) > 0:
            bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        else:
            bboxes = []

        return keypoints, bboxes

    def __body_orient(self, ls, rs, lh, rh):
        """Compute torso orientation features.

        Uses the center point between left/right shoulders and left/right hips
        to compute an angle in degrees, plus a simple height/width ratio.

        Args:
            ls: Left shoulder point (x, y) in nose-relative coordinates.
            rs: Right shoulder point (x, y) in nose-relative coordinates.
            lh: Left hip point (x, y) in nose-relative coordinates.
            rh: Right hip point (x, y) in nose-relative coordinates.

        Returns:
            tuple[float, float]: (angle_degrees, height_width_ratio)
        """
        shoulder_center = (ls + rs) / 2
        hip_center = (lh + rh) / 2
        dx = hip_center[0] - shoulder_center[0]
        dy = hip_center[1] - shoulder_center[1]
        angles = np.degrees(np.arctan2(dy, dx))

        width = np.abs(ls[0] - rh[0])
        height = np.abs(ls[1] - rh[1])
        ratio = height / (width + 1e-6)

        return angles, ratio
    
    def __bounding_box_ratio(self, p1, p2):
        """Compute bounding-box height/width ratio.

        Args:
            p1: Top-left corner (x1, y1).
            p2: Bottom-right corner (x2, y2).

        Returns:
            float: height / (width + 1e-6)
        """
        [x1, y1], [x2, y2] = p1, p2

        width = np.abs(x1 - x2)
        height = np.abs(y1 - y2)
        ratio = height / (width + 1e-6)
        return ratio

    def detect_fall(self, image):
        """Detect whether each detected person in an image is fallen.

        For each detected person, this method builds a feature vector from:
        - Torso angle + torso height/width ratio
        - Bounding-box height/width ratio
        - Mean of all keypoint y-values (nose-relative)

        Args:
            image: Path to an image (or a source supported by
                Ultralytics' `YOLO.predict`).

        Returns:
            list[tuple(str, bbox)]: One label per detected person, each either `'fallen'` or
            `'not-fallen'` and bounding box according to the image.
        """
        keypoints, bboxes = self.__get_points(image)

        inference = []
        for points, bbox in zip(keypoints, bboxes):
            nose = points[0]
            new_point = list(map(lambda p: self.__nose_ref_point(p, nose), points))
            
            # Extract the features point
            left_shoulder = np.asarray(new_point[5])
            right_shoulder = np.asarray(new_point[6])
            left_hip = np.asarray(new_point[11])
            right_hip = np.asarray(new_point[12])
            bbox_p1 = np.asarray(bbox[:2])
            bbox_p2 = np.asarray(bbox[2:])

            all_y = []
            for ele in new_point:
                all_y.append(ele[1])
            mean = np.mean(all_y) # Calculate mean of all points

            angle_body = np.array(self.__body_orient(left_shoulder, right_shoulder, left_hip, right_hip))
            bbox_ratio = self.__bounding_box_ratio(bbox_p1, bbox_p2)
            features = np.append(angle_body, [bbox_ratio, mean]).reshape(1, -1) # Resize with 2D array
            
            scaled_features = self.scaler.transform(features)
            prediction = self.svm.predict(scaled_features) # Predict using SVC
            
            if prediction[0] == 0:
                inference.append(('not-fallen', bbox))
            else:
                inference.append(('fallen', bbox))
        return inference
