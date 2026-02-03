import cv2

from src.childClassifier import ChildModel
from src.fallDetection import FallDetect
from src.behaviorAnalysis import BehaviorAnalysisModel
from utilities.utils import Utils
from utilities.logger import logging

logger = logging.getLogger(__name__)

class SystemPipeline:
    def __init__(self, child_model_path, fall_detect_path, behavior_model_path, video_path=None):
        self.video_path = video_path
        self.child_model_path = child_model_path
        self.fall_detect_path = fall_detect_path
        self.behavior_model_path = behavior_model_path

        # Read video from path or real-time camera
        if self.video_path:
            self.video = cv2.VideoCapture(self.video_path)
            logger.info('Video are feed from the file path')
        else:
            self.video = cv2.VideoCapture(0)
            logger.info('Video are feed from real time camera')
        
        # Raise an erros
        if not self.video.isOpened():
            raise ValueError('Error: Could not open video file.')
        
        # Child model instance
        self.childModel = ChildModel(self.child_model_path, conf=0.8)
        logger.info('Child Model are loaded')

        # Fall detection instance
        self.fallDetection = FallDetect(self.fall_detect_path)
        logger.info('Falll Detection model is getting loaded')

        # Behavioral anaylysis model instance
        self.behaviorModel = BehaviorAnalysisModel(self.behavior_model_path, conf=0.8)
        logger.info('Behavioral Model is loaded')

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
                    logger.error('Error: while extracting the frame')
                    break

                count = 0
                frame_infos = {}

                # Retrieve child info from the frame
                childs = self.childModel.child_track(frame)
                logger.info('Children are detected')
                for child in childs:
                    [track_id, cls, conf, bbox] = child
                    info = {
                        'f_result': [],
                        'f_bbox': [],
                        'b_result': [],
                        'b_bbox': []
                    }

                    # Filter the child from the frame
                    image_child = Utils.crop_image(frame, bbox=bbox)
                    logger.info('Children are extracted from the frame')
                    
                    # Detect the fall
                    fall_results = self.fallDetection.detect_fall(image_child)
                    for r in fall_results:
                        info['f_result'].append(r[0])
                        info['f_bbox'].append(r[1])
                    logger.info('Fallen children are recognized')

                    # Behavior analysis
                    behavioral_results = self.behaviorModel.detect_behavior(image_child)
                    for r in behavioral_results:
                        info['b_result'].append(self.behaviorModel.get_names()[int(r[0])])
                        info['b_bbox'].append(r[2])
                    logger.info('Extracted the behavior')

                    frame_infos[track_id] = info
                    logger.info('Added the frame info to the dict')

                for info in frame_infos:
                    ## TODO: update the frame
                    pass


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break