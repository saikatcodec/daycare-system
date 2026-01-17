import cv2
from childClassifier import ChildModel
from utilities.utils import Utils

class SystemPipeline:
    def __init__(self, child_model_path, video_path=None):
        self.video_path = video_path
        self.child_model_path = child_model_path

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

                # Retrieve child info from the frame
                childs = self.childModel.child_track(frame)
                ## TODO: rest of the tasks

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break