# Pose Perfect

A real-time pose recognition game that uses webcam-based human pose detection and a trained machine learning model to challenge players to match randomly selected poses within a timed game session.

## Highlights

* Uses MediaPipe Pose to detect body landmarks from a live webcam feed.
* Uses a trained Keras model with `labels.npy` to classify detected poses.
* Provides an interactive Pygame interface with multiple game states, scoring, timers, and replay functionality.
* Starts each game with a 60-second timer and tracks score, high score, and average time per move.
* Includes pose reference images, custom fonts, backgrounds, and game-over screens as bundled assets.
* Supports webcam backend selection for Windows and Linux environments.
* Includes PyInstaller configuration and Windows build scripts for generating `PoseMaster.exe`.

## Project Structure

* `Game.py`: Main game application, webcam processing, pose prediction, game loop, scoring, and UI.
* `model.h5`: Trained Keras pose classification model.
* `labels.npy`: Pose labels used to map model predictions to pose names.
* `poseImages/`: Reference images displayed for the target poses.
* `Images/`: UI assets such as title, clock, visibility warning, and game-over graphics.
* `Background_Images/`: Background images used throughout the game.
* `Fonts/`: Custom fonts used by the Pygame interface.
* `PoseMaster.spec`: PyInstaller specification for packaging the game.
* `build_windows.bat`: Windows command-line build script.
* `build_windows.ps1`: PowerShell build script.
* `high_score.txt`: Persistent high-score storage.

## Quick Start

Create a virtual environment and install the required libraries:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install opencv-python numpy mediapipe pygame tensorflow keras
```

Run the game:

```bash
python Game.py
```

Make sure a working webcam is available before starting the game. The application opens the camera through OpenCV and processes the live feed with MediaPipe.

## Verify the Project

To verify that the main application starts correctly:

```bash
python Game.py
```

To build the Windows executable:

```bat
build_windows.bat
```

Or with PowerShell:

```powershell
.\build_windows.ps1
```

The provided build configuration uses PyInstaller and produces `PoseMaster.exe`.

## Game Flow

The application is organized around three main states:

* `home`: Main menu with Start and Quit options.
* `start`: Webcam preview, pose detection, and current high score.
* `game`: Timed pose-matching gameplay with score tracking.

After the game ends, the player can view the final score, high score, and time-per-move statistics, then either play again or return to the main menu.

## Pose Recognition

For each webcam frame, MediaPipe extracts pose landmarks. The application converts the landmarks into relative coordinates, feeds the resulting feature vector into the trained Keras model, and selects the label with the highest prediction score.

The game randomly selects a target pose and checks the model's prediction against that target during gameplay.

## Notes

* The game runs at a `1240 × 700` Pygame window and targets approximately 30 FPS.
* A webcam is required for pose detection.
* The application includes platform-specific OpenCV camera backends to improve camera compatibility.
* The trained model and label files are loaded from the project resources at runtime.
* High scores are persisted locally through `high_score.txt`.

## Build Notes

The repository is set up to package the game together with its model, labels, fonts, images, backgrounds, and pose assets. The Windows build scripts automatically install or update PyInstaller and invoke `PoseMaster.spec`.
