import cv2
from cv2.typing import MatLike

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
        
        # Raise an errors
        if not self.video.isOpened():
            raise ValueError('Error: Could not open video file.')
        
        # Child model instance
        self.childModel = ChildModel(self.child_model_path, conf=0.8)
        logger.info('Child Model are loaded')

        # Fall detection instance
        self.fallDetection = FallDetect(self.fall_detect_path)
        logger.info('Fall Detection model is getting loaded')

        # Behavioral analysis model instance
        self.behaviorModel = BehaviorAnalysisModel(self.behavior_model_path, conf=0.8)
        logger.info('Behavioral Model is loaded')


    def __infer_frame(self, frame: MatLike) ->  dict[int, dict]:
        frame_infos: dict[int, dict] = {} # To store the frame info

        # Retrieve child info from the frame
        children = self.childModel.child_track(frame)
        logger.info('Children are detected')
        for child in children:
            [track_id, cls, conf, bbox] = child
            info = {
                'child_bbox': bbox
            }

            # Filter the child from the frame
            image_child = Utils.crop_image(frame, bbox=bbox)
            logger.info('Children are extracted from the frame')
            
            # Detect the fall
            fall_results = self.fallDetection.detect_fall(image_child)
            for r in fall_results:
                info['f_result'] = r[0]
                info['f_bbox'] = r[1]
            logger.info('Fallen children are recognized')

            # Behavior analysis
            behavioral_results = self.behaviorModel.detect_behavior(image_child)
            for r in behavioral_results:
                info['b_result'] = self.behaviorModel.get_names()[int(r[0])]
                info['b_bbox'] = r[2]
            logger.info('Extracted the behavior')

            frame_infos[track_id] = info
            logger.info('Added the frame info to the dict')

        return frame_infos


    def __update_frame(self, frame: MatLike, frame_infos: dict[int, dict]) -> MatLike:
        for track_id, info in frame_infos.items():
            # If fall or behavior detect draw bounding box with them
            if info.get('f_result', None) == 'fallen':
                # Draw bbox
                frame = Utils.draw_bounding_box(
                    frame,
                    info['f_result'],
                    info['f_bbox'],
                    color=(24, 31, 230) # In BGR mode
                )
                logger.info('Draw a bounding box for the fall detection')

            if info.get('b_result', None):
                # Draw the bbox
                frame = Utils.draw_bounding_box(
                    frame,
                    info['b_result'],
                    info['b_bbox'],
                    color=(244, 31, 20) # In BGR mode
                )
                logger.info('Draw a bounding box for behavior analysis')
            #  Otherwise draw the bounding box for children
            else:
                frame = Utils.draw_bounding_box(
                    frame,
                    f'Child id: {track_id}',
                    info['child_bbox'],
                    color=(112, 0, 94) # In BGR mode
                )
                logger.info('Child detection box added')

        return frame


    def start(self):
        skip_frame = 5
        count = 0

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
                # Get the frame information
                frame_infos = self.__infer_frame(frame)
                # Update the frame according to the information
                frame = self.__update_frame(frame, frame_infos)

                # show the window or frame
                cv2.imshow('ChildVision', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break