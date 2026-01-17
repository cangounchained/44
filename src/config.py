"""
Configuration module for gaze tracking research application.

This module defines all constants, paths, and hyperparameters used
throughout the gaze tracking system.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# CAMERA & MEDIAPIPE SETTINGS
# =============================================================================
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# MediaPipe Face Mesh indices
# Right eye: 33, 160, 158, 133, 153, 144
# Left eye: 362, 385, 387, 373, 380, 381
# Iris: right_iris=[468, 469, 470, 471, 472], left_iris=[473, 474, 475, 476, 477]

RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
LEFT_EYE_LANDMARKS = [362, 385, 387, 373, 380, 381]

RIGHT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
LEFT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]

# Facial ROI indices for attention distribution
NOSE_TIP = 1
LEFT_EYE_CENTER = 159  # Left eye approximate center
RIGHT_EYE_CENTER = 386  # Right eye approximate center
MOUTH_CENTER = 13  # Mouth approximate center

# =============================================================================
# STIMULUS SETTINGS
# =============================================================================
STIMULUS_RADIUS = 15  # pixels
STIMULUS_COLOR = (0, 0, 255)  # BGR: Red

# Stimulus movement patterns
STIMULUS_SPEED = 2.0  # pixels per frame
STIMULUS_CHANGE_INTERVAL = 120  # frames before changing direction
STIMULUS_DWELL_TIME = 60  # frames stimulus stays in one location

# =============================================================================
# GAZE TRACKING THRESHOLDS
# =============================================================================
EAR_THRESHOLD = 0.18  # Eye Aspect Ratio threshold for blink detection
BLINK_CONSEC_FRAMES = 3  # Minimum consecutive frames to classify as blink

# Fixation detection: gaze variance threshold
FIXATION_VARIANCE_THRESHOLD = 50.0  # pixels^2
FIXATION_MIN_DURATION = 150  # milliseconds (approx frames/30)

# Saccade detection
SACCADE_VELOCITY_THRESHOLD = 100  # degrees per second

# Gaze smoothing
GAZE_SMOOTHING_FACTOR = 0.6  # EMA factor for gaze position

# =============================================================================
# FEATURE EXTRACTION WINDOWS
# =============================================================================
WINDOW_SIZE_MS = 2000  # 2 seconds for feature aggregation
WINDOW_SIZE_FRAMES = int(WINDOW_SIZE_MS / (1000 / CAMERA_FPS))

# =============================================================================
# ASD MODELING PARAMETERS
# =============================================================================
MODEL_TYPE = "randomforest"  # Options: "randomforest", "neural_network"

# RandomForest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_RANDOM_STATE = 42

# Neural Network hyperparameters (if using PyTorch)
NN_HIDDEN_SIZE = 64
NN_DROPOUT = 0.3
NN_LEARNING_RATE = 0.001
NN_EPOCHS = 50
NN_BATCH_SIZE = 32

# =============================================================================
# SYNTHETIC TRAINING DATA
# =============================================================================
N_SYNTHETIC_SAMPLES = 500
N_ASD_POSITIVE = 250
N_ASD_NEGATIVE = 250

# Synthetic data characteristics
# Typical ASD-associated patterns:
# - Higher fixation durations on facial features
# - Lower gaze switching rate
# - Reduced attention to eyes
# - Reduced tracking accuracy for moving stimuli
# - Higher blink rate variation

SYNTHETIC_PARAMS_HEALTHY = {
    "fixation_duration_mean": 200,  # ms
    "fixation_duration_std": 50,
    "gaze_switch_rate": 4.0,  # switches per second
    "gaze_switch_std": 0.8,
    "eye_attention": 0.35,  # proportion of fixations on eyes
    "mouth_attention": 0.20,
    "nose_attention": 0.15,
    "off_face_attention": 0.30,
    "tracking_accuracy": 0.85,
    "blink_rate": 15,  # blinks per minute
    "blink_rate_std": 3,
}

SYNTHETIC_PARAMS_ASD = {
    "fixation_duration_mean": 350,
    "fixation_duration_std": 100,
    "gaze_switch_rate": 2.0,
    "gaze_switch_std": 0.6,
    "eye_attention": 0.15,
    "mouth_attention": 0.25,
    "nose_attention": 0.30,
    "off_face_attention": 0.30,
    "tracking_accuracy": 0.65,
    "blink_rate": 22,
    "blink_rate_std": 5,
}

# =============================================================================
# SCORING & PERCENTILE SETTINGS
# =============================================================================
SCORE_MIN = 0
SCORE_MAX = 100

# Percentile bins for risk classification
PERCENTILE_LOW = 33
PERCENTILE_MODERATE = 67
# Elevated: > PERCENTILE_MODERATE

# =============================================================================
# UI & VISUALIZATION SETTINGS
# =============================================================================
SHOW_GAZE_VECTOR = True
SHOW_FACIAL_LANDMARKS = True
SHOW_EYE_LANDMARKS = True
SHOW_STIMULUS = True
SHOW_ROI_BOXES = True

# Colors (BGR format for OpenCV)
COLOR_GAZE_VECTOR = (0, 255, 0)  # Green
COLOR_BLINK_INDICATOR = (0, 0, 255)  # Red
COLOR_FIXATION = (255, 255, 0)  # Cyan
COLOR_LANDMARK = (255, 0, 0)  # Blue
COLOR_EYE = (0, 165, 255)  # Orange

# =============================================================================
# LOGGING & OUTPUT SETTINGS
# =============================================================================
LOG_LEVEL = "INFO"
SAVE_VIDEOS = False
EXPORT_PLOTS = True

# =============================================================================
# ETHICAL & DISCLAIMERS
# =============================================================================
DISCLAIMER_TEXT = """
⚠️ RESEARCH & EDUCATIONAL USE ONLY

This tool analyzes gaze patterns associated with autism spectrum disorder (ASD). 

IMPORTANT LIMITATIONS:
• This is NOT a diagnostic system
• Results are statistical pattern analysis only
• Should NOT replace professional assessment
• Individual variation is significant
• No clinical or medical claims should be made
• For research and educational purposes only

Use this tool responsibly and ethically.
"""

# =============================================================================
# RESEARCH METADATA
# =============================================================================
RESEARCH_TITLE = "Real-Time Gaze Tracking Analysis: Behavioral Patterns & Statistical Modeling"
RESEARCH_VERSION = "1.0.0"
RESEARCH_AFFILIATION = "Neurodevelopmental Research Lab"

