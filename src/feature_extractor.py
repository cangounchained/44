"""
Feature extraction module for gaze tracking data.

Computes behavioral metrics from raw gaze tracking data over time windows:
- Fixation characteristics
- Saccade metrics
- Blink statistics
- Attention distribution across ROIs
- Gaze dispersion and stability
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from src.gaze_tracker import GazeData
from src.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    WINDOW_SIZE_FRAMES,
    FIXATION_VARIANCE_THRESHOLD,
    FIXATION_MIN_DURATION,
    SACCADE_VELOCITY_THRESHOLD,
    CAMERA_FPS
)


@dataclass
class GazeFeatures:
    """Container for extracted gaze features."""
    
    # Fixation metrics
    fixation_duration_mean: float  # ms
    fixation_duration_max: float
    fixation_duration_std: float
    fixation_count: int
    
    # Gaze switching
    gaze_switch_rate: float  # switches per second
    saccade_count: int
    saccade_velocity_mean: float  # deg/sec
    saccade_velocity_max: float
    saccade_amplitude_mean: float  # degrees
    
    # Blink metrics
    blink_rate: float  # blinks per minute
    blink_count: int
    blink_duration_mean: float  # ms
    
    # Gaze stability
    gaze_dispersion: float  # variance in gaze position (pixels^2)
    gaze_velocity_mean: float  # pixels per frame
    gaze_velocity_std: float
    gaze_entropy: float  # spatial entropy of gaze distribution
    
    # Eye metrics
    left_eye_aspect_ratio_mean: float
    right_eye_aspect_ratio_mean: float
    pupil_asymmetry_mean: float  # left-right distance
    
    # Attention distribution (proportion of fixations)
    attention_left_eye: float
    attention_right_eye: float
    attention_nose: float
    attention_mouth: float
    attention_off_face: float
    
    # Tracking performance
    stimulus_tracking_accuracy: float  # How closely gaze follows stimulus
    gaze_latency_ms: float  # Delay between stimulus and gaze
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        return np.array(list(asdict(self).values()))
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names in order."""
        return list(asdict(GazeFeatures(
            fixation_duration_mean=0,
            fixation_duration_max=0,
            fixation_duration_std=0,
            fixation_count=0,
            gaze_switch_rate=0,
            saccade_count=0,
            saccade_velocity_mean=0,
            saccade_velocity_max=0,
            saccade_amplitude_mean=0,
            blink_rate=0,
            blink_count=0,
            blink_duration_mean=0,
            gaze_dispersion=0,
            gaze_velocity_mean=0,
            gaze_velocity_std=0,
            gaze_entropy=0,
            left_eye_aspect_ratio_mean=0,
            right_eye_aspect_ratio_mean=0,
            pupil_asymmetry_mean=0,
            attention_left_eye=0,
            attention_right_eye=0,
            attention_nose=0,
            attention_mouth=0,
            attention_off_face=0,
            stimulus_tracking_accuracy=0,
            gaze_latency_ms=0,
        )).keys())


class FeatureExtractor:
    """
    Extract behavioral features from gaze tracking data.
    
    Maintains a rolling window of gaze data and computes aggregate features.
    """
    
    def __init__(self, window_size_frames: int = WINDOW_SIZE_FRAMES):
        """
        Initialize feature extractor.
        
        Args:
            window_size_frames: Number of frames to accumulate before computing features
        """
        self.window_size = window_size_frames
        self.gaze_buffer = deque(maxlen=window_size_frames)
        
        # Stimulus tracking
        self.stimulus_position = None
        self.stimulus_history = deque(maxlen=window_size_frames)
        
        # Fixation tracking
        self.current_fixation_start = None
        self.current_fixation_position = None
        self.fixations = []
        self.blinks = deque(maxlen=100)
        
        # Frame counter
        self.frame_count = 0
    
    def add_frame(
        self,
        gaze_data: GazeData,
        stimulus_position: Tuple[float, float] = None
    ) -> None:
        """
        Add a new frame of gaze data to the buffer.
        
        Args:
            gaze_data: Gaze data for this frame
            stimulus_position: (x, y) position of stimulus in pixels, if available
        """
        self.gaze_buffer.append(gaze_data)
        
        if stimulus_position is not None:
            self.stimulus_history.append(stimulus_position)
        
        # Update fixation tracking
        self._update_fixation(gaze_data)
        
        # Update blink tracking
        if gaze_data.is_blinking:
            if not self.blinks or self.frame_count - self.blinks[-1][1] > 10:
                self.blinks.append((self.frame_count, self.frame_count))
            else:
                # Extend current blink
                self.blinks[-1] = (self.blinks[-1][0], self.frame_count)
        
        self.frame_count += 1
    
    def _update_fixation(self, gaze_data: GazeData) -> None:
        """Track fixations (periods of stable gaze)."""
        if not gaze_data.face_detected:
            if self.current_fixation_start is not None:
                self._end_fixation()
            return
        
        if self.current_fixation_start is None:
            # Start new fixation
            self.current_fixation_start = self.frame_count
            self.current_fixation_position = gaze_data.gaze_point_2d.copy()
        else:
            # Check if gaze position is stable
            variance = np.sum((gaze_data.gaze_point_2d - self.current_fixation_position) ** 2)
            
            if variance > FIXATION_VARIANCE_THRESHOLD:
                # Fixation broken
                self._end_fixation()
                self.current_fixation_start = self.frame_count
                self.current_fixation_position = gaze_data.gaze_point_2d.copy()
    
    def _end_fixation(self) -> None:
        """End current fixation and log it."""
        if self.current_fixation_start is None:
            return
        
        duration_frames = self.frame_count - self.current_fixation_start
        duration_ms = duration_frames * (1000 / CAMERA_FPS)
        
        # Only record fixations longer than minimum duration
        if duration_ms >= FIXATION_MIN_DURATION:
            self.fixations.append({
                'start_frame': self.current_fixation_start,
                'end_frame': self.frame_count,
                'duration_ms': duration_ms,
                'position': self.current_fixation_position.copy()
            })
        
        self.current_fixation_start = None
        self.current_fixation_position = None
    
    def extract_features(self) -> GazeFeatures:
        """
        Extract behavioral features from current buffer.
        
        Returns:
            GazeFeatures: Computed features
        """
        if len(self.gaze_buffer) == 0:
            return self._get_empty_features()
        
        # Fixation features
        fix_durations = [f['duration_ms'] for f in self.fixations]
        fixation_duration_mean = np.mean(fix_durations) if fix_durations else 0
        fixation_duration_max = np.max(fix_durations) if fix_durations else 0
        fixation_duration_std = np.std(fix_durations) if fix_durations else 0
        
        # Gaze switching
        gaze_switch_rate = self._compute_gaze_switch_rate()
        saccade_metrics = self._compute_saccade_metrics()
        
        # Blink metrics
        blink_rate = len(self.blinks) * 60 / max(self.frame_count / CAMERA_FPS, 1)
        blink_duration_mean = (
            np.mean([b[1] - b[0] for b in self.blinks]) * (1000 / CAMERA_FPS)
            if self.blinks else 0
        )
        
        # Gaze stability
        gaze_positions = np.array([g.gaze_point_2d for g in self.gaze_buffer])
        gaze_dispersion = self._compute_gaze_dispersion(gaze_positions)
        gaze_velocities = self._compute_gaze_velocities(gaze_positions)
        gaze_entropy = self._compute_gaze_entropy(gaze_positions)
        
        # Eye metrics
        ear_left = [g.left_eye_aspect_ratio for g in self.gaze_buffer]
        ear_right = [g.right_eye_aspect_ratio for g in self.gaze_buffer]
        pupil_asym = [g.pupil_distance_lr for g in self.gaze_buffer]
        
        # Attention distribution
        attention = self._compute_attention_distribution()
        
        # Stimulus tracking
        tracking_accuracy, latency_ms = self._compute_tracking_performance()
        
        return GazeFeatures(
            fixation_duration_mean=float(fixation_duration_mean),
            fixation_duration_max=float(fixation_duration_max),
            fixation_duration_std=float(fixation_duration_std),
            fixation_count=len(self.fixations),
            gaze_switch_rate=float(gaze_switch_rate),
            saccade_count=saccade_metrics['count'],
            saccade_velocity_mean=float(saccade_metrics['velocity_mean']),
            saccade_velocity_max=float(saccade_metrics['velocity_max']),
            saccade_amplitude_mean=float(saccade_metrics['amplitude_mean']),
            blink_rate=float(blink_rate),
            blink_count=len(self.blinks),
            blink_duration_mean=float(blink_duration_mean),
            gaze_dispersion=float(gaze_dispersion),
            gaze_velocity_mean=float(np.mean(gaze_velocities) if gaze_velocities else 0),
            gaze_velocity_std=float(np.std(gaze_velocities) if gaze_velocities else 0),
            gaze_entropy=float(gaze_entropy),
            left_eye_aspect_ratio_mean=float(np.mean(ear_left) if ear_left else 0),
            right_eye_aspect_ratio_mean=float(np.mean(ear_right) if ear_right else 0),
            pupil_asymmetry_mean=float(np.mean(pupil_asym) if pupil_asym else 0),
            attention_left_eye=float(attention.get('left_eye', 0)),
            attention_right_eye=float(attention.get('right_eye', 0)),
            attention_nose=float(attention.get('nose', 0)),
            attention_mouth=float(attention.get('mouth', 0)),
            attention_off_face=float(attention.get('off_face', 0)),
            stimulus_tracking_accuracy=float(tracking_accuracy),
            gaze_latency_ms=float(latency_ms),
        )
    
    def _compute_gaze_switch_rate(self) -> float:
        """Compute number of gaze switches per second."""
        if len(self.fixations) < 2:
            return 0.0
        
        total_duration_s = max(self.frame_count / CAMERA_FPS, 1)
        # Each fixation represents a gaze position; switches are transitions
        switches = max(len(self.fixations) - 1, 0)
        return switches / total_duration_s
    
    def _compute_saccade_metrics(self) -> Dict:
        """Compute saccade (rapid eye movement) metrics."""
        gaze_positions = np.array([g.gaze_point_2d for g in self.gaze_buffer])
        
        if len(gaze_positions) < 3:
            return {
                'count': 0,
                'velocity_mean': 0.0,
                'velocity_max': 0.0,
                'amplitude_mean': 0.0
            }
        
        # Compute velocities
        velocities = np.linalg.norm(np.diff(gaze_positions, axis=0), axis=1)
        
        # Detect saccades (high velocity movements)
        saccade_threshold = np.percentile(velocities, 75) if len(velocities) > 0 else 0
        saccade_mask = velocities > saccade_threshold
        
        saccade_count = np.sum(saccade_mask)
        saccade_velocities = velocities[saccade_mask] if saccade_count > 0 else np.array([0])
        
        # Compute amplitudes from fixation positions
        amplitudes = []
        for i in range(len(self.fixations) - 1):
            pos1 = self.fixations[i]['position']
            pos2 = self.fixations[i + 1]['position']
            dist = np.linalg.norm(pos2 - pos1)
            # Convert pixels to approximate degrees (typical: 1 degree ≈ 50-100 pixels)
            amplitude_deg = dist / 75.0
            amplitudes.append(amplitude_deg)
        
        return {
            'count': int(saccade_count),
            'velocity_mean': float(np.mean(saccade_velocities)),
            'velocity_max': float(np.max(saccade_velocities)),
            'amplitude_mean': float(np.mean(amplitudes) if amplitudes else 0)
        }
    
    def _compute_gaze_dispersion(self, gaze_positions: np.ndarray) -> float:
        """Compute variance in gaze positions."""
        if len(gaze_positions) < 2:
            return 0.0
        
        center = np.mean(gaze_positions, axis=0)
        distances_sq = np.sum((gaze_positions - center) ** 2, axis=1)
        return float(np.mean(distances_sq))
    
    def _compute_gaze_velocities(self, gaze_positions: np.ndarray) -> List[float]:
        """Compute frame-to-frame gaze velocities."""
        if len(gaze_positions) < 2:
            return []
        
        velocities = np.linalg.norm(np.diff(gaze_positions, axis=0), axis=1)
        return velocities.tolist()
    
    def _compute_gaze_entropy(self, gaze_positions: np.ndarray) -> float:
        """
        Compute spatial entropy of gaze distribution.
        Higher entropy indicates more dispersed attention.
        """
        if len(gaze_positions) < 10:
            return 0.0
        
        # Create 2D histogram of gaze positions
        hist, _, _ = np.histogram2d(
            gaze_positions[:, 0], gaze_positions[:, 1],
            bins=10, range=[[0, CAMERA_WIDTH], [0, CAMERA_HEIGHT]]
        )
        
        # Compute entropy
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-6)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist + 1e-6))
        
        return float(entropy)
    
    def _compute_attention_distribution(self) -> Dict[str, float]:
        """
        Compute proportion of attention to different facial regions.
        Based on fixation locations.
        """
        if len(self.fixations) == 0:
            return {
                'left_eye': 0.2,
                'right_eye': 0.2,
                'nose': 0.2,
                'mouth': 0.2,
                'off_face': 0.2
            }
        
        h, w = CAMERA_HEIGHT, CAMERA_WIDTH
        
        # Define ROI bounds (in pixels)
        left_eye_region = (w * 0.25, w * 0.4, h * 0.3, h * 0.45)
        right_eye_region = (w * 0.6, w * 0.75, h * 0.3, h * 0.45)
        nose_region = (w * 0.4, w * 0.6, h * 0.35, h * 0.55)
        mouth_region = (w * 0.3, w * 0.7, h * 0.55, h * 0.7)
        
        attention = {
            'left_eye': 0, 'right_eye': 0, 'nose': 0,
            'mouth': 0, 'off_face': 0
        }
        
        for fixation in self.fixations:
            pos = fixation['position']
            x, y = pos[0], pos[1]
            
            if left_eye_region[0] <= x <= left_eye_region[1] and \
               left_eye_region[2] <= y <= left_eye_region[3]:
                attention['left_eye'] += 1
            elif right_eye_region[0] <= x <= right_eye_region[1] and \
                 right_eye_region[2] <= y <= right_eye_region[3]:
                attention['right_eye'] += 1
            elif nose_region[0] <= x <= nose_region[1] and \
                 nose_region[2] <= y <= nose_region[3]:
                attention['nose'] += 1
            elif mouth_region[0] <= x <= mouth_region[1] and \
                 mouth_region[2] <= y <= mouth_region[3]:
                attention['mouth'] += 1
            else:
                attention['off_face'] += 1
        
        # Normalize to proportions
        total = sum(attention.values())
        if total > 0:
            attention = {k: v / total for k, v in attention.items()}
        else:
            attention = {k: 0.2 for k in attention.keys()}
        
        return attention
    
    def _compute_tracking_performance(self) -> Tuple[float, float]:
        """
        Compute stimulus tracking accuracy and latency.
        
        Returns:
            (accuracy, latency_ms): Accuracy [0-1] and latency in milliseconds
        """
        if len(self.stimulus_history) < 10 or len(self.gaze_buffer) < 10:
            return 0.5, 0.0
        
        # Convert stimulus history to array
        stimulus_pos = np.array(list(self.stimulus_history))
        gaze_pos = np.array([g.gaze_point_2d for g in self.gaze_buffer])
        
        # Ensure same length
        min_len = min(len(stimulus_pos), len(gaze_pos))
        stimulus_pos = stimulus_pos[-min_len:]
        gaze_pos = gaze_pos[-min_len:]
        
        # Compute accuracy as inverse of normalized distance
        distances = np.linalg.norm(stimulus_pos - gaze_pos, axis=1)
        diagonal = np.sqrt(CAMERA_WIDTH ** 2 + CAMERA_HEIGHT ** 2)
        normalized_distances = distances / diagonal
        accuracy = float(np.clip(1.0 - np.mean(normalized_distances), 0, 1))
        
        # Compute latency using cross-correlation
        if len(distances) > 30:
            correlations = []
            for lag in range(-15, 16):
                if lag < 0:
                    corr = np.corrcoef(stimulus_pos[:-abs(lag), 0], gaze_pos[abs(lag):, 0])[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(stimulus_pos[lag:, 0], gaze_pos[:-lag, 0])[0, 1]
                else:
                    corr = np.corrcoef(stimulus_pos[:, 0], gaze_pos[:, 0])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            
            best_lag = np.argmax(correlations) - 15
            latency_ms = best_lag * (1000 / CAMERA_FPS)
        else:
            latency_ms = 0.0
        
        return accuracy, float(latency_ms)
    
    def reset(self) -> None:
        """Reset feature extractor for new session."""
        self.gaze_buffer.clear()
        self.stimulus_history.clear()
        self.fixations.clear()
        self.blinks.clear()
        self.current_fixation_start = None
        self.frame_count = 0
    
    @staticmethod
    def _get_empty_features() -> GazeFeatures:
        """Return features with all zeros."""
        return GazeFeatures(
            fixation_duration_mean=0, fixation_duration_max=0,
            fixation_duration_std=0, fixation_count=0,
            gaze_switch_rate=0, saccade_count=0,
            saccade_velocity_mean=0, saccade_velocity_max=0,
            saccade_amplitude_mean=0, blink_rate=0,
            blink_count=0, blink_duration_mean=0,
            gaze_dispersion=0, gaze_velocity_mean=0,
            gaze_velocity_std=0, gaze_entropy=0,
            left_eye_aspect_ratio_mean=0, right_eye_aspect_ratio_mean=0,
            pupil_asymmetry_mean=0, attention_left_eye=0,
            attention_right_eye=0, attention_nose=0,
            attention_mouth=0, attention_off_face=0,
            stimulus_tracking_accuracy=0, gaze_latency_ms=0
        )

