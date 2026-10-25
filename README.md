# Pose Master

> The Maven artifact and Java entry class are still named `pose-perfect` / `PosePerfectApp`.
> This codebase builds on the [NotMayankfr/Pose-Perfect](https://github.com/NotMayankfr/Pose-Perfect) repository (MIT); its commit history is preserved here.

A real-time pose recognition game that uses webcam-based human pose detection and a trained machine learning model to challenge players to match randomly selected poses within a timed game session.

## Highlights

* Uses MediaPipe BlazePose models (person detector + pose landmarker) running on ONNX Runtime to detect body landmarks from a live webcam feed.
* Uses a trained pose-classification model exported from the original Keras `model.h5` to ONNX, with `labels.npy` for class names.
* Provides an interactive Java Swing interface with home, webcam preview, gameplay, and game-over screens.
* Starts each game with a 60-second timer and tracks score, high score, and average time per move.
* Includes pose reference images, custom fonts, backgrounds, and game-over screens as bundled assets.
* Supports webcam backend selection for Windows (DSHOW/MSMF), macOS (AVFoundation), and Linux (V4L2).
* No Python runtime is required — the application is a self-contained Java desktop game.

## Technology Stack

| Layer | Library |
|-------|---------|
| Language | Java 17+ |
| Build | Maven |
| GUI | Java Swing |
| Webcam | OpenCV 4.9 (via org.openpnp:opencv) |
| Pose detection | ONNX Runtime 1.17.1 running MediaPipe BlazePose models |
| Pose classification | ONNX Runtime running the exported Keras MLP classifier |
| Labels | Direct `.npy` parsing (no NumPy dependency) |
| Testing | JUnit 5 |

## Project Structure

```
Pose-Master/
├── pom.xml                          # Maven build configuration
├── README.md                        # This file
├── model.h5                         # Original Keras model (kept for reference)
├── labels.npy                       # Pose class labels
├── high_score.txt                   # Bundled seed high score
├── src/main/java/com/posemaster/    # Java source code
│   ├── PosePerfectApp.java          # Application entry point
│   ├── app/                         # HighScoreStore, ResourceLocator
│   ├── camera/                      # WebcamCapture (OpenCV)
│   ├── game/                        # GameSession (rules, scoring, timing)
│   ├── ml/                          # OnnxPoseClassifier, PoseLabels, PoseClassifier
│   ├── pose/                        # MediapipePoseDetector, PoseLandmarks, BlazePoseAnchors
│   └── ui/                          # GameFrame, Assets, Button, TextOverlay
├── src/main/resources/              # Bundled assets (on classpath)
│   ├── models/pose_classifier.onnx  # Exported classifier (ONNX)
│   ├── models/person_detection_mediapipe_2023mar.onnx
│   ├── models/pose_estimation_mediapipe_2023mar.onnx
│   ├── labels.npy
│   ├── poseImages/*.png             # Reference images for each pose
│   ├── Background_Images/*.png
│   ├── Images/*.png                 # UI assets (title, clock, gameover, etc.)
│   └── Fonts/*.ttf
├── src/test/java/com/posemaster/    # JUnit 5 tests
│   ├── app/HighScoreStoreTest.java
│   ├── game/GameSessionTest.java
│   ├── ml/OnnxPoseClassifierTest.java
│   ├── ml/PoseLabelsTest.java
│   ├── pose/MediapipePoseDetectorTest.java
│   └── tools/SmokeTest.java
├── src/test/resources/              # Test fixtures (pose.jpg, expected_landmarks.json)
├── models/                          # Copies of the two MediaPipe ONNX models used by tools/reference_pipeline.py
│   └── README.md                    # Notes on the person-detection model's origin
├── docs/diagrams/                   # Mermaid (.mmd) architecture, data-flow, state, sequence and threading diagrams
├── Game.py, PoseMaster.spec, build_windows.*   # Original Python version (reference only)
├── Background_Images/, Fonts/, Images/, poseImages/   # Asset copies used by the Python version
└── tools/                           # Conversion/verification scripts (not needed to run the game)
    ├── convert_model_to_onnx.py     # Exports model.h5 to ONNX
    ├── verify_onnx.py               # Verifies ONNX parity with Keras
    ├── reference_pipeline.py        # Python reference for pose pipeline comparison
    └── requirements-*.txt           # Python deps for tools only (not runtime)
```

## Quick Start

### Requirements

* Java JDK 17 or newer (build verified with JDK 23.0.1)
* Apache Maven 3.6 or newer (build verified with Maven 3.9.9)
* A working webcam

The ONNX models are bundled in `src/main/resources/models/` and are packaged into the JAR; nothing needs to be downloaded at runtime. Maven downloads dependencies on the first build.

### Build

```bash
mvn clean package
```

This compiles the application, runs the test suite, and produces a self-contained shaded JAR at `target/pose-perfect.jar`.

### Run

```bash
java -jar target/pose-perfect.jar
```

### Running tests

```bash
mvn test
```

The suite has 22 JUnit 5 tests (high-score storage, game rules, ONNX classifier, label parsing, and pose detection on a fixture image). They need no webcam.

### Webcam selection

The application tries platform-appropriate capture backends in order:

* Windows: `CAP_DSHOW`, `CAP_MSMF`, `CAP_ANY`
* macOS: `CAP_AVFOUNDATION`, `CAP_ANY`
* Linux: `CAP_V4L2`, `CAP_ANY`

If no webcam is available, the game still launches but pose detection and classification will not work until a camera is connected.

## Game Flow

The application has four screens:

1. **Home** — Main menu with "Start" and "Quit" buttons.
2. **Start** — Webcam preview with current high score. Press "Start" to begin or "Back" to return home.
3. **Game** — Timed pose-matching gameplay (60 seconds, +1.5s per match). The target pose changes every 7 seconds or immediately after a match.
4. **Game Over** — Shows final score, time played, high score, and average time per move. Press "Play Again" or "Main Menu".

## Pose Recognition Pipeline

1. OpenCV captures a frame from the webcam.
2. The MediaPipe person detector (`person_detection_mediapipe_2023mar.onnx`) locates the person and returns a bounding box plus four body keypoints.
3. The region of interest is cropped, rotated upright, and resized to 256×256.
4. The MediaPipe pose landmarker (`pose_estimation_mediapipe_2023mar.onnx`) predicts 33 pose landmarks with visibility.
5. Landmarks are converted to a 66-value feature vector (`(x - x0, y - y0)` per landmark relative to the nose).
6. The ONNX classifier (`pose_classifier.onnx`) predicts the pose label.
7. The game checks the prediction against the current target pose.

## Model Conversion

The original Keras model (`model.h5`) was exported to ONNX using `tools/convert_model_to_onnx.py`. The exported graph reproduces the Keras predictions exactly (verified by `tools/verify_onnx.py`). The Python tooling is only needed for conversion/verification and is **not** a runtime dependency.

## High Scores

High scores are persisted to `~/.pose_master/high_score.txt`, seeded from the bundled `high_score.txt` on first run.

## Known Limitations

* Automated tests and the build do not exercise a live webcam. Real-time gameplay with a camera has not been verified by these checks.
* The build and tests were run on Windows only; the macOS/Linux capture backends listed above are untested.
* No classifier accuracy figures are provided.

## Troubleshooting

* **`mvn` not found / `JAVA_HOME is not defined correctly`** — install Maven and point `JAVA_HOME` at a JDK 17+ install.
* **"No webcam detected"** — close other apps using the camera, check OS camera permissions, and reconnect the device.
* **Poor pose detection** — keep your whole body in frame with good lighting.

## Notes

* The game window is `1240 × 700` and targets approximately 30 FPS.
* A webcam is required for pose detection; without one, the game shows a "No webcam detected" message.
* Camera capture and inference run on a background thread so the Swing UI never freezes.
* The original Python implementation (`Game.py`, PyInstaller spec, build scripts) is retained in the repository for reference but is not used by the Java application.