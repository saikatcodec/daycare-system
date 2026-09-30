# ChildVision: Intelligent Child Day-Care Monitoring System

ChildVision is a Python-based computer vision project designed to monitor children in a daycare or classroom environment in real time. It combines object detection, tracking, pose-based fall detection, and behavior analysis to identify potentially unsafe or abnormal events such as falls, crying, and fighting.

The system is built with Ultralytics YOLO models, OpenCV, NumPy, and scikit-learn, and is structured as a modular inference pipeline that can process a video file or a live camera feed.

## Project Overview

The project aims to provide intelligent supervision for daycare environments by detecting:

- Children in the scene
- Falls or sudden posture changes
- Crying or distress signals
- Aggressive interactions such as fighting

This is especially useful for passive monitoring, early alerts, and safety support in childcare centers where staff may need assistance detecting incidents in real time.

## Core Goals

- Detect and track children continuously across frames
- Identify child fall events using human pose estimation
- Recognize suspicious or unsafe behavior patterns
- Visualize detections with bounding boxes and labels
- Support real-time video analytics from a file or camera
- Save annotated output video for review

## System Architecture

The system is organized into a small set of modules:

- `main.py` — application entry point
- `src/childClassifier.py` — child detection and tracking logic
- `src/fallDetection.py` — fall detection using pose keypoints and SVM classification
- `src/behaviorAnalysis.py` — cry/fight detection through YOLO object recognition
- `src/pipeline.py` — orchestration of the full monitoring workflow
- `utilities/logger.py` — centralized logging system
- `models/` — trained model checkpoints and preprocessing artifacts

## How the Pipeline Works

The full pipeline follows a multi-stage approach:

1. Child detection and tracking on the full frame
   - Uses YOLO-based detection with BoT-SORT tracking
   - Tracks each child across successive frames
   - Maintains consistent IDs for each child

2. Behavior detection on the full frame
   - Runs YOLO inference for cry and fight detection
   - Full-frame context is important because multi-child interactions need the entire scene

3. Fall detection on each child crop
   - Creates a crop around each detected child
   - Uses YOLO pose estimation to get keypoints
   - Extracts torso-related posture features
   - Classifies fall vs non-fall using a trained SVM model

4. Annotation and output
   - Draws child boxes, fall alerts, and behavior alerts on the video frames
   - Displays the processed feed in a window
   - Optionally saves output as a video file

## Main Modules

### 1. Child Detection and Tracking

File: `src/childClassifier.py`

This module loads a YOLO child detection model and tracks children with BoT-SORT. It returns a list of tracked children with:

- track ID
- class index
- confidence score
- bounding box coordinates

This is the foundation of the surveillance system because all later decisions depend on identifying each child consistently.

### 2. Fall Detection

File: `src/fallDetection.py`

This module performs pose-based fall detection. It:

- loads a YOLO pose model
- extracts keypoints for each child
- normalizes keypoints relative to the nose landmark
- computes a feature vector based on torso angle, torso ratio, bounding box ratio, and vertical displacement
- scales the features using a saved scaler
- predicts whether the child has fallen using an SVM model

This is a classic human activity recognition approach built around keypoint geometry rather than raw image-only classification.

### 3. Behavioral Analysis

File: `src/behaviorAnalysis.py`

This module loads a YOLO behavior model to detect:

- crying
- fighting

It returns each detection with its label, confidence, and bounding box. These events are drawn directly onto the full video frame.

### 4. Pipeline Orchestration

File: `src/pipeline.py`

This module ties everything together. It:

- opens the video source
- runs child tracking on each frame
- launches fall detection in parallel for each child crop
- runs behavioral analysis once per frame
- draws the results on the annotated video
- keeps the system responsive in real time

## Project Structure

```text
daycare-system/
├── README.md
├── main.py
├── requirements.txt
├── .gitignore
├── .gitattributes
├── models/
│   ├── behavioralModel.pt
│   ├── behavioralModelv2.pt
│   ├── childModel.pt
│   ├── fallDetect.pkl
│   ├── fallDetectScaler.pkl
│   ├── fall_detection_v1.3.0.pkl
│   ├── scaler_v1.2.1.pkl
│   └── yolo11m-pose.pt
├── src/
│   ├── __init__.py
│   ├── behaviorAnalysis.py
│   ├── childClassifier.py
│   ├── fallDetection.py
│   ├── pipeline.py
│   └── __init__.py
├── utilities/
│   ├── __init__.py
│   ├── logger.py
│   └── utils.py
├── tests/
│   └── test_note.ipynb
├── logs/
│   └── (generated at runtime)
├── data/
│   └── (user-provided input video, if used)
├── outputs/
│   └── (generated output videos)
└── .vscode/
    └── settings.json
```

## Requirements

The project depends on:

- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics
- NumPy
- scikit-learn
- pandas
- matplotlib
- seaborn
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/saikatcodec/daycare-system.git
cd daycare-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your media and model files

This project expects trained model files under the `models/` directory. If the required checkpoints or pickle files are not present, the application will raise a `FileNotFoundError`.

The main script also points to a sample input video:

```python
video_path = Path("data/input_2.mp4")
output_path = Path("outputs/output_2.mp4")
```

If you want to run it on a live camera instead, change `video_path` to `None` or use a webcam device index.

### 4. Run the system

```bash
python main.py
```

Press `q` in the OpenCV window to stop the application.

## Configuration Notes

The main script sets the following defaults:

```python
child_model_path = Path("models/childModel.pt")
fall_detect_path = Path("models/fallDetect.pkl")
behavior_model_path = Path("models/behavioralModelv2.pt")
video_path = Path("data/input_2.mp4")
output_path = Path("outputs/output_2.mp4")
```

You can modify these paths to suit your environment or replace the provided example media with your own footage.

## Output Behavior

When a child or event is detected, the pipeline annotates the output video with highlighted boxes and labels such as:

- Child ID
- `fallen`
- `cry`
- `fight`

Warnings are also written to the logger, which is stored in `logs/` for debugging and review.

## Strengths of the Project

- Modular design for separate detection tasks
- Real-time monitoring capability
- Strong use of state-of-the-art vision models
- Clear separation between child tracking, fall detection, and behavior analysis
- Supports both file-based and camera-based input

## Limitations

- Requires pretrained model artifacts to be available locally
- Performance depends on hardware capabilities, especially GPU support
- Behavior detection may vary depending on camera quality, lighting, and clutter
- Fall detection is based on selected pose features and may need tuning for different environments
- The project is primarily a prototype and research-oriented implementation

## Potential Improvements

- Add CLI arguments instead of hardcoded paths
- Add configuration file support (`YAML` or `JSON`)
- Add event logging to a database or dashboard
- Improve alerting via email, SMS, or notifications
- Support multi-camera processing
- Add a web dashboard for live monitoring
- Train and fine-tune custom models for daycare-specific conditions

## Use Cases

This system can be used in:

- daycare centers
- kindergarten classrooms
- child monitoring systems
- school safety observation tools
- smart surveillance research applications

## Conclusion

ChildVision is a practical AI-based surveillance prototype designed to enhance daycare safety through intelligent video analysis. It integrates multiple computer vision components into a single monitoring pipeline and demonstrates a practical use of deep learning for child safety and behavior awareness.

The project is well suited for further development into a production-grade childcare monitoring solution with better configuration management, model versioning, and real-time alert systems.

## License

This repository does not currently declare a license in the project metadata. If you plan to distribute or share the project publicly, it is recommended to add an appropriate open-source license such as MIT or Apache-2.0.

## Author

Project repository: `saikatcodec/daycare-system`

---

If you want, I can also create a more polished version of this README tailored for GitHub, including badges, a project architecture diagram, and a demo screenshot section.
