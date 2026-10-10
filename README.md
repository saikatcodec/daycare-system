# ChildVision: Intelligent Child Daycare Monitoring System

ChildVision is a Python-based computer vision system designed for real-time monitoring in daycare and classroom environments. It combines object detection, object tracking, pose estimation, and behavioral analysis to support safer supervision and early incident detection.

The project uses Ultralytics YOLO, OpenCV, NumPy, PyTorch, and scikit-learn to process either recorded video or live camera input. Its modular pipeline is intended for research, prototype deployment, and educational use in AI-driven childcare supervision.

> Note: This project is a research and prototype system. It is not a substitute for human supervision, childcare staff oversight, or emergency response procedures.

## Key Features

- Detects and tracks children across video frames
- Assigns stable IDs to each detected child
- Identifies falls using pose-based feature analysis and SVM classification
- Detects crying and fighting using a YOLO behavior model
- Draws bounding boxes, labels, and alerts directly on processed frames
- Supports both file-based and camera-based video input
- Saves annotated output videos for review and analysis
- Logs warnings and runtime information for debugging

## System Architecture

The repository is organized into the following core components:

- `main.py` — application entry point and runtime configuration
- `src/childClassifier.py` — child detection and tracking logic
- `src/fallDetection.py` — pose-based fall detection using keypoints and an SVM model
- `src/behaviorAnalysis.py` — crying and fighting detection
- `src/pipeline.py` — end-to-end monitoring workflow
- `utilities/logger.py` — centralized logging
- `utilities/utils.py` — reusable support routines
- `models/` — trained model checkpoints and preprocessing assets

## Pipeline Overview

The monitoring pipeline is designed as a multi-stage process:

1. Child detection and tracking
   - Detects children with a YOLO-based model
   - Tracks each child across frames using BoT-SORT
   - Maintains stable IDs, confidence scores, and bounding boxes

2. Behavior analysis
   - Runs a full-scene YOLO inference pass for cry and fight detection
   - Uses overall frame context to interpret multi-child interactions

3. Fall detection
   - Extracts a crop around each detected child
   - Estimates pose keypoints using a YOLO pose model
   - Computes posture statistics such as torso angle, torso ratio, and displacement
   - Classifies the child as fallen or not fallen using a trained SVM

4. Annotation and output
   - Draws boxes, IDs, and alert labels over each frame
   - Displays the processed feed in an OpenCV window
   - Optionally saves annotated output to a video file

## Project Structure

```text
daycare-system/
├── README.md
├── LICENSE
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
├── data/              # User-provided input video files
├── logs/              # Generated runtime logs
├── outputs/           # Generated annotated output videos
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

### 3. Prepare model and media files

This project expects the required trained model artifacts in the `models/` directory. If the required files are missing, the application will raise a `FileNotFoundError`.

The default input and output paths are configured in `main.py` as:

```python
video_path = Path("data/input_2.mp4")
output_path = Path("outputs/output_2.mp4")
```

Update these paths to match your local files, or switch to a live camera source in the project code.

### 4. Run the application

```bash
python main.py
```

Press `q` in the OpenCV window to stop the processing loop.

## Configuration Notes

The main runtime configuration includes the following defaults:

```python
child_model_path = Path("models/childModel.pt")
fall_detect_path = Path("models/fallDetect.pkl")
behavior_model_path = Path("models/behavioralModelv2.pt")
video_path = Path("data/input_2.mp4")
output_path = Path("outputs/output_2.mp4")
```

These values can be adjusted to point to different models, media sources, or output locations.

## Output Behavior

When a child or event is detected, the pipeline annotates the output frame with labels such as:

- Child ID
- `fallen`
- `cry`
- `fight`

Warnings and runtime activity are written to the `logs/` directory for review and debugging.

## Intended Use Cases

This system is suitable for:

- Daycare and kindergarten environments
- Child monitoring and safety observation
- Classroom supervision support
- Research in intelligent video analytics
- Early-stage AI-driven safety monitoring prototypes

## Limitations

- Requires locally available pretrained model artifacts.
- Performance depends on hardware capability, especially GPU availability.
- Accuracy may vary due to camera quality, lighting, scene clutter, and occlusion.
- Fall detection relies on selected pose features and may need tuning for different environments.
- The project remains a prototype and is not a certified commercial safety system.

## Potential Improvements

- Add command-line arguments for configuration
- Add YAML or JSON configuration support
- Improve event storage and reporting
- Add notification integrations such as email or SMS
- Support multi-camera input
- Develop a web dashboard for live monitoring
- Continue model training for daycare-specific conditions

## License

This project is licensed under the Prosperity Public License 3.0.0.

The source code is publicly available for academic, educational, and
non-commercial research purposes. Commercial use is subject to the
terms and limitations of the Prosperity Public License, including its
commercial trial period.

For commercial use beyond the permitted trial period, please contact
the copyright holder for licensing options.

See the [LICENSE](LICENSE.md) file for the complete terms and conditions.

## Author

Project repository: [saikatcodec/daycare-system](https://github.com/saikatcodec/daycare-system)
