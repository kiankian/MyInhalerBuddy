## Project Overview

MyInhalerBuddy is a computer vision application that helps users properly use inhalers by providing real-time feedback through webcam monitoring. The application uses OpenCV, MediaPipe, and NumPy to detect red inhalers, track facial landmarks, and analyze proper inhaler technique across multiple specialized modes.

## Architecture

The codebase follows a mode-based architecture pattern:

- **Main Application (`InhalerBuddyApp`)**: Manages mode switching, webcam capture, and UI overlay
- **Mode System**: Abstract base classes `AppMode` and `BaseProcessingMode` provide framework for specialized functionality
- **Computer Vision Pipeline**: Core detection functions for red inhaler tracking and facial analysis
- **Tracking System**: `InhalerTracker` class provides temporal consistency and prediction for inhaler position

### Key Components

**Detection & Tracking (`main.py:181-350`)**:
- `detect_red_inhaler()`: HSV color space filtering with shape analysis
- `InhalerTracker`: Motion prediction and temporal filtering to maintain tracking during occlusion
- `calculate_inhaler_distance_to_lips()`: Spatial relationship analysis between detected objects

**Face Analysis (`main.py:352-429`)**:
- `calculate_head_pitch()`: Head orientation analysis using MediaPipe landmarks
- `draw_lips()`: Lip visualization and landmark highlighting
- Integration with MediaPipe Face Mesh for facial feature detection

**Mode Architecture (`main.py:432-894`)**:
- `BaseProcessingMode`: Common processing pipeline shared across modes
- Specialized modes: FullMode, HeadPositioningMode, InhalerShakingMode, etc.
- Each mode can override `process_head_pitch()` and `process_custom()` for specific functionality

## Development Environment

**Python Version**: 3.11.10 (specified in `.python-version`)

**Key Dependencies**:
- opencv-contrib-python 4.8.1.78 (computer vision)
- mediapipe 0.10.21 (facial landmark detection)
- numpy 1.24.3 (numerical computing)
- sounddevice 0.5.2 (audio processing capabilities)

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Run the main application
python main.py
```

## Key Usage Patterns

**Mode Switching**: Press keys 1-8 to switch between different testing modes
**Debug Visualization**: Press 'm' in any mode to toggle red color mask display
**Camera Controls**: Application automatically uses camera index 0 with horizontal flip for mirror effect

## Important Implementation Details

**Color Detection**: Uses dual HSV range detection for red objects with adaptive thresholds based on tracking state
**Tracking Persistence**: InhalerTracker maintains detection across 15 frames of gaps using motion prediction
**Head Pitch Calculation**: Uses nose bridge to nose tip ratio compared to bridge-to-chin distance, calibrated for 90° = straight ahead
**Mode Pattern**: Each mode inherits common CV pipeline but can customize specific analysis steps

## Development Notes

The application uses a single-file architecture where all functionality is contained in `main.py`. The mode system allows for focused testing of individual inhaler technique aspects while sharing common computer vision infrastructure.
