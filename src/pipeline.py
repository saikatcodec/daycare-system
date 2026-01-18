import cv2
from childClassifier import ChildModel
from fallDetection import FallDetect
from behaviorAnalysis import BehaviorAnalysisModel
from utilities.utils import Utils

class SystemPipeline:
    def __init__(self, child_model_path, fall_detect_path, behavior_model_path,video_path=None):
        self.video_path = video_path
        self.child_model_path = child_model_path
        self.fall_detect_path = fall_detect_path
        self.behavior_model_path = behavior_model_path

        # Read video from path or real-time camera
        if self.video_path:
            self.video = cv2.VideoCapture(self.video_path)
        else:
            self.video = cv2.VideoCapture(0)
        
        # Raise an erros
        if not self.video.isOpened():
            raise ValueError('Error: Could not open video file.')
        
        # Child model instance
        self.childModel = ChildModel(self.child_model_path, conf=0.8)
        # Fall detection instance
        self.fallDetection = FallDetect(self.fall_detect_path)
        # Behavioral anaylysis model instance
        self.behaviorModel = BehaviorAnalysisModel(self.behavior_model_path, conf=0.8)

    def start(self):
        skip_frame = 5, count = 0

        # Extract frame after every 5 frames
        while True:
            ret = self.video.grab()
            if not ret:
                print("End of video or error.")
                break

            count += 1
            if count % skip_frame == 0:
                ret, frame = self.video.retrieve()
                if not ret:
                    print("Error: while extracting the frame")
                    break

                count = 0
                frame_info = []

                # Retrieve child info from the frame
                childs = self.childModel.child_track(frame)
                for child in childs:
                    info = {}
                    [track_id, cls, conf, bbox] = child
                    info[track_id] = {}
                    # Filter the child from the frame
                    image_child = Utils.crop_image(frame, bbox=bbox)
                    
                    # Detect the fall
                    fall_results = self.fallDetection.detect_fall(image_child)
                    for r in fall_results:
                        info[track_id]['f_result'] = r[0]
                        info[track_id]['f_bbox'] = r[1]

                    # Behavior analysis
                    behavioral_results = self.behaviorModel.detect_behavior(image_child)
                    for r in behavioral_results:
                        info[track_id]['b_result'] = self.behaviorModel.get_names()[int(r[0])]
                        info[track_id]['b_bbox'] = r[2]

                    frame_info.append(info)
                ## TODO: update the frame


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break