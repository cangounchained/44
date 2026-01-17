"""
Initialization module for gaze tracking package.
"""

from src.config import *
from src.gaze_tracker import GazeTracker, GazeData, GazeVisualizer
from src.feature_extractor import FeatureExtractor, GazeFeatures
from src.stimulus import StimulusRenderer, StimulusPattern
from src.model import GazePatternModel, SyntheticDataGenerator, ModelPrediction

__version__ = "1.0.0"
__author__ = "Neurodevelopmental Research Lab"

__all__ = [
    "GazeTracker",
    "GazeData",
    "GazeVisualizer",
    "FeatureExtractor",
    "GazeFeatures",
    "StimulusRenderer",
    "StimulusPattern",
    "GazePatternModel",
    "SyntheticDataGenerator",
    "ModelPrediction"
]
