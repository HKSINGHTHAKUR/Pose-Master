# Third-party notices

Pose Master's own code is under the MIT Licence (see [LICENSE](LICENSE)). It includes or depends on the material below. Licence information is taken from the files themselves (font metadata, Maven POMs, `models/README.md`) where noted; where the repository does not record it, that is stated.

## Inherited from the original Pose-Perfect repository

The Python game (`Game.py`), `model.h5`, `labels.npy`, the artwork (`Images/`, `Background_Images/`, `poseImages/`) and the fonts were first published in [NotMayankfr/Pose-Perfect](https://github.com/NotMayankfr/Pose-Perfect) under the MIT Licence. Its copyright notice is kept in [LICENSE](LICENSE), and its commit history is part of this repository's history.

## Models

| File | Source and licence |
|------|--------------------|
| `person_detection_mediapipe_2023mar.onnx` | MediaPipe Pose person detector converted to ONNX. [models/README.md](models/README.md) (the model's own readme) states: Apache 2.0, based on [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) and [MediaPipe](https://github.com/google/mediapipe). |
| `pose_estimation_mediapipe_2023mar.onnx` | MediaPipe pose landmarker in ONNX form. The repository does not record where this file came from. Its name follows the OpenCV Zoo naming scheme and MediaPipe is Apache 2.0, but this has not been confirmed. |
| `pose_classifier.onnx` | Generated from `model.h5` by `tools/convert_model_to_onnx.py`. |

Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>

## Fonts (`Fonts/`, `src/main/resources/Fonts/`)

| Font | Licence, as recorded in the font file |
|------|---------------------------------------|
| Marcellus (c) 2012 Brian J. Bonislawsky DBA Astigmatic (AOETI) | SIL Open Font License 1.1 |
| Nabla (c) 2022 The Nabla Project Authors | SIL Open Font License 1.1 |
| Silkscreen (c) 2001 The Silkscreen Project Authors | SIL Open Font License 1.1 |
| `god-of-war.ttf` ("GodOfWar") | No licence or author is recorded in the font file or the repository. Check its terms before redistributing. |

SIL OFL 1.1: <https://openfontlicense.org>

## Artwork

The images in `Images/`, `Background_Images/` and `poseImages/` (and their copies in `src/main/resources/`) have no recorded source or licence beyond the original repository's MIT Licence. Confirm that you may redistribute any image that was not created for this project.

## Libraries (downloaded by Maven, bundled into the shaded JAR)

| Library | Licence (from its POM) |
|---------|------------------------|
| OpenCV 4.9.0 via `org.openpnp:opencv:4.9.0-0` | BSD licence per the POM; OpenCV 4.5 and later is Apache 2.0 |
| ONNX Runtime 1.17.1 (`com.microsoft.onnxruntime`) | MIT |
| JUnit Jupiter 5.10.2 (tests only, not bundled) | Eclipse Public License 2.0 |

The Python scripts in `tools/` have their own dependencies, listed in `tools/requirements-*.txt`; they are not distributed with the game.
