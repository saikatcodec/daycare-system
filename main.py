import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.pipeline import SystemPipeline

if __name__=="__main__":
    # Define the paths for the models and video
    child_model_path = 'models/childModel.pt'
    fall_detect_path = 'models/fallDetect.pkl'
    behavior_model_path = 'models/behavioralModel.pt'
    video_path = 'data/input_1.mp4'  # Set to None for real-time camera feed

    # Create an instance of the SystemPipeline
    pipeline = SystemPipeline(child_model_path, fall_detect_path, behavior_model_path, video_path)

    # Start processing the video feed
    pipeline.start()
    

