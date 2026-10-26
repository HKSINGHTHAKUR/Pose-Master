# Third-party notices

Pose Master's own code is under the MIT Licence (see [LICENSE](LICENSE)). It includes or depends on the material below. Licence information is taken from the files themselves (font metadata, Maven POMs, the model publisher) where noted; where the repository does not record it, that is stated.

## Models

| File | Source and licence |
|------|--------------------|
| `person_detection_mediapipe_2023mar.onnx` | MediaPipe Pose person detector converted to ONNX. Its publisher (OpenCV Zoo) states: Apache 2.0, based on [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) and [MediaPipe](https://github.com/google/mediapipe). |
| `pose_estimation_mediapipe_2023mar.onnx` | MediaPipe pose landmarker in ONNX form. The repository does not record where this file came from. Its name follows the OpenCV Zoo naming scheme and MediaPipe is Apache 2.0, but this has not been confirmed. |
| `pose_classifier.onnx` | The game's trained pose classifier, stored as ONNX. |

Apache License 2.0: <https://www.apache.org/licenses/LICENSE-2.0>

## Fonts (`src/main/resources/Fonts/`)

| Font | Licence, as recorded in the font file |
|------|---------------------------------------|
| Marcellus (c) 2012 Brian J. Bonislawsky DBA Astigmatic (AOETI) | SIL Open Font License 1.1 |
| Nabla (c) 2022 The Nabla Project Authors | SIL Open Font License 1.1 |
| Silkscreen (c) 2001 The Silkscreen Project Authors | SIL Open Font License 1.1 |
| `god-of-war.ttf` ("GodOfWar") | No licence or author is recorded in the font file or the repository. Check its terms before redistributing. |

SIL OFL 1.1: <https://openfontlicense.org>

## Artwork

The images in `src/main/resources/` (`Images/`, `Background_Images/` and `poseImages/`) are distributed under the project's MIT Licence.

## Libraries (downloaded by Maven, bundled into the shaded JAR)

| Library | Licence (from its POM) |
|---------|------------------------|
| OpenCV 4.9.0 via `org.openpnp:opencv:4.9.0-0` | BSD licence per the POM; OpenCV 4.5 and later is Apache 2.0 |
| ONNX Runtime 1.17.1 (`com.microsoft.onnxruntime`) | MIT |
| JUnit Jupiter 5.10.2 (tests only, not bundled) | Eclipse Public License 2.0 |
