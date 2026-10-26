# Pose Master

A real-time pose-matching game. The webcam watches you, a neural network recognises which of 16 poses you are holding, and you score points by matching the pose shown on screen before the timer runs out.

The game is a Java 17 Swing desktop application. Pose detection and classification run locally through ONNX Runtime, so no network connection is needed to play.

## How it plays

1. **Home**: choose *Start* or *Quit*.
2. **Start**: webcam preview and the current high score; choose *Start* to begin or *Back* to return.
3. **Game**: a target pose is shown. The session lasts 60 seconds, and every matched pose adds 1.5 seconds. The target changes after 7 seconds, or immediately after a match. The body must be fully in frame (both ankles and both wrists detected with visibility of at least 0.6) before a pose is classified.
4. **Game over**: final score, time played, high score, and average time per move; choose *Play Again* or *Main Menu*.

The high score is stored in `~/.pose_master/high_score.txt` and starts at 0 when that file does not exist. The 16 poses are listed in `src/main/resources/labels.txt`; reference images are in `src/main/resources/poseImages/`.

## Architecture

```
webcam frame (OpenCV)
  -> person detector (MediaPipe, ONNX)          -> body box + keypoints
  -> crop, rotate upright, resize to 256x256
  -> pose landmarker (MediaPipe, ONNX)          -> 33 landmarks with visibility
  -> 66-value feature vector                    -> (x, y) of every landmark relative to the nose
  -> classifier (Dense 66-128-64-16, ONNX)      -> pose label from labels.txt
  -> GameSession                                -> score, timers, target pose
```

Camera capture and inference run on a background thread so the Swing UI stays responsive. The capture backends tried are `CAP_DSHOW`, `CAP_MSMF` and `CAP_ANY` on Windows, `CAP_AVFOUNDATION` and `CAP_ANY` on macOS, and `CAP_V4L2` and `CAP_ANY` on Linux. The window is 1240 x 700 and targets about 30 FPS. If no camera opens, the game still launches and shows a "No webcam detected" message.

Diagrams of the architecture, data flow, game states, call sequence and threading are in `docs/diagrams/` (Mermaid `.mmd` files).

## Technology stack

| Layer | Technology |
|-------|-----------|
| Language | Java 17+ |
| Build | Maven (shaded, runnable JAR) |
| GUI | Java Swing |
| Camera and image processing | OpenCV 4.9.0 (`org.openpnp:opencv:4.9.0-0`) |
| Inference | ONNX Runtime 1.17.1 |
| Pose models | MediaPipe BlazePose person detector and pose landmarker (ONNX) |
| Classifier | Dense neural network (ONNX) |
| Labels | `labels.txt`, one label per line |
| Tests | JUnit 5.10.2 |

## Build and run

Requirements: JDK 17 or newer, Apache Maven 3.6 or newer, and a webcam to play.

```bash
mvn clean package          # compile, run the tests, build target/pose-master.jar
java -jar target/pose-master.jar
mvn test                   # tests only
```

The build was verified with JDK 23.0.1 and Maven 3.9.9. Maven downloads dependencies on the first build; the models and assets are bundled in the JAR and nothing is downloaded at runtime.

## Models and runtime assets

All of these live in `src/main/resources/` and are packaged into the JAR:

| File | Purpose |
|------|---------|
| `models/pose_classifier.onnx` | Pose classifier |
| `models/person_detection_mediapipe_2023mar.onnx` | MediaPipe person detector |
| `models/pose_estimation_mediapipe_2023mar.onnx` | MediaPipe pose landmarker |
| `labels.txt` | Names of the 16 poses |
| `poseImages/`, `Images/`, `Background_Images/`, `Fonts/` | Artwork and fonts |

## Project structure

```
Pose-Master/
├── pom.xml
├── src/main/java/com/posemaster/
│   ├── PoseMasterApp.java           # entry point
│   ├── app/                         # HighScoreStore, ResourceLocator
│   ├── camera/                      # WebcamCapture
│   ├── game/                        # GameSession (rules, scoring, timing)
│   ├── ml/                          # OnnxPoseClassifier, PoseLabels, PoseClassifier
│   ├── pose/                        # MediapipePoseDetector, PoseLandmarks, BlazePoseAnchors
│   └── ui/                          # GameFrame, Assets, Button, TextOverlay
├── src/main/resources/              # models, labels and artwork
├── src/test/java/com/posemaster/    # JUnit tests, plus tools/SmokeTest (a manual headless check)
├── src/test/resources/              # pose.jpg and expected_landmarks.json fixtures
├── docs/diagrams/                   # Mermaid diagrams
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

## Tests

`mvn test` runs 22 JUnit tests covering the high-score file, the game rules and timers, the ONNX classifier, label parsing, and pose detection on a fixture image. None of them needs a camera.

## Known limitations

* The tests and build do not use a live webcam. Real-time play with a camera has not been verified by them.
* The build and tests were run on Windows only. The macOS and Linux capture backends are untested.
* No classifier accuracy figures are provided.
* The game always opens camera index 0; there is no camera selection.

## Troubleshooting

* **`mvn` not found, or "JAVA_HOME is not defined correctly"**: install Maven and point `JAVA_HOME` at a JDK 17+ installation.
* **"No webcam detected"**: close other programs that use the camera, check the operating system's camera permissions, and reconnect the device.
* **Poses are not recognised**: keep your whole body, including ankles and wrists, in frame with good lighting.

## Licence and third-party notices

The project is released under the MIT Licence; see [LICENSE](LICENSE).

Models, fonts, libraries and artwork from other sources, with their licences, are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
