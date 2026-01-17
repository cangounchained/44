"""
Core gaze tracking module using MediaPipe Face Mesh.

This module handles:
- Real-time facial landmark detection
- Eye and iris tracking
- 3D gaze vector estimation
- Blink detection
- Pupil-based gaze direction
"""

import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Tuple, List
from src.config import (
    RIGHT_EYE_LANDMARKS, LEFT_EYE_LANDMARKS,
    RIGHT_IRIS_LANDMARKS, LEFT_IRIS_LANDMARKS,
    EAR_THRESHOLD, BLINK_CONSEC_FRAMES,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    GAZE_SMOOTHING_FACTOR
)


@dataclass
class GazeData:
    """Container for gaze tracking data at a single frame."""
    timestamp: float
    gaze_point_3d: np.ndarray  # [x, y, z] in normalized space
    gaze_point_2d: np.ndarray  # [x, y] in pixel space
    gaze_vector: np.ndarray    # [x, y, z] normalized direction vector
    left_eye_center: np.ndarray
    right_eye_center: np.ndarray
    left_eye_aspect_ratio: float
    right_eye_aspect_ratio: float
    is_blinking: bool
    pupil_left: np.ndarray
    pupil_right: np.ndarray
    pupil_distance_lr: float  # Left-right pupil distance (asymmetry metric)
    head_pose: np.ndarray  # [pitch, yaw, roll] in degrees
    landmarks_3d: Optional[np.ndarray] = None
    face_detected: bool = False


class GazeTracker:
    """
    Real-time gaze tracker using MediaPipe Face Mesh.
    
    Features:
    - 468 facial landmarks with 3D coordinates
    - Iris detection for pupil tracking
    - Eye aspect ratio for blink detection
    - Smooth gaze vector estimation
    """
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh detector."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.blink_counter_left = 0
        self.blink_counter_right = 0
        self.prev_gaze_point = None
        self.gaze_history = []
        
    def process_frame(self, frame: np.ndarray) -> GazeData:
        """
        Process a single frame and extract gaze data.
        
        Args:
            frame: RGB image frame (H, W, 3)
            
        Returns:
            GazeData: Gaze information for this frame
        """
        h, w, c = frame.shape
        gaze_data = GazeData(
            timestamp=0,
            gaze_point_3d=np.array([0.5, 0.5, 0.0]),
            gaze_point_2d=np.array([w/2, h/2]),
            gaze_vector=np.array([0.0, 0.0, 1.0]),
            left_eye_center=np.array([0.0, 0.0]),
            right_eye_center=np.array([0.0, 0.0]),
            left_eye_aspect_ratio=0.0,
            right_eye_aspect_ratio=0.0,
            is_blinking=False,
            pupil_left=np.array([0.0, 0.0]),
            pupil_right=np.array([0.0, 0.0]),
            pupil_distance_lr=0.0,
            head_pose=np.array([0.0, 0.0, 0.0]),
            face_detected=False
        )
        
        # Convert to RGB if necessary
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame.dtype == np.uint8:
                # Assume BGR format from OpenCV
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
        else:
            return gaze_data
        
        # Run face mesh detection
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return gaze_data
        
        landmarks = results.multi_face_landmarks[0]
        gaze_data.face_detected = True
        
        # Extract 3D landmarks
        landmarks_3d = np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks.landmark
        ])
        gaze_data.landmarks_3d = landmarks_3d
        
        # Compute eye aspect ratio (EAR) for blink detection
        left_ear = self._compute_eye_aspect_ratio(landmarks_3d, LEFT_EYE_LANDMARKS)
        right_ear = self._compute_eye_aspect_ratio(landmarks_3d, RIGHT_EYE_LANDMARKS)
        
        gaze_data.left_eye_aspect_ratio = left_ear
        gaze_data.right_eye_aspect_ratio = right_ear
        
        # Detect blink
        is_left_blinking = left_ear < EAR_THRESHOLD
        is_right_blinking = right_ear < EAR_THRESHOLD
        
        self.blink_counter_left = (self.blink_counter_left + 1) if is_left_blinking else 0
        self.blink_counter_right = (self.blink_counter_right + 1) if is_right_blinking else 0
        
        gaze_data.is_blinking = (
            self.blink_counter_left >= BLINK_CONSEC_FRAMES or 
            self.blink_counter_right >= BLINK_CONSEC_FRAMES
        )
        
        # Get eye centers
        left_eye_center = np.mean(landmarks_3d[LEFT_EYE_LANDMARKS], axis=0)
        right_eye_center = np.mean(landmarks_3d[RIGHT_EYE_LANDMARKS], axis=0)
        
        gaze_data.left_eye_center = left_eye_center[:2]
        gaze_data.right_eye_center = right_eye_center[:2]
        
        # Get pupil (iris) centers
        left_iris_center = np.mean(landmarks_3d[LEFT_IRIS_LANDMARKS], axis=0)
        right_iris_center = np.mean(landmarks_3d[RIGHT_IRIS_LANDMARKS], axis=0)
        
        gaze_data.pupil_left = left_iris_center[:2]
        gaze_data.pupil_right = right_iris_center[:2]
        
        # Compute pupil asymmetry (left-right distance)
        pupil_distance = np.linalg.norm(left_iris_center - right_iris_center)
        gaze_data.pupil_distance_lr = pupil_distance
        
        # Estimate gaze direction from iris position relative to eye center
        gaze_vector_left = self._compute_gaze_direction(
            left_eye_center, left_iris_center
        )
        gaze_vector_right = self._compute_gaze_direction(
            right_eye_center, right_iris_center
        )
        
        # Average the two eye gaze vectors
        gaze_vector = (gaze_vector_left + gaze_vector_right) / 2.0
        gaze_vector = gaze_vector / (np.linalg.norm(gaze_vector) + 1e-6)
        
        gaze_data.gaze_vector = gaze_vector
        
        # Estimate 2D gaze point on screen
        gaze_point_2d = self._estimate_2d_gaze_point(
            left_eye_center, right_eye_center,
            left_iris_center, right_iris_center,
            landmarks_3d, h, w
        )
        
        # Apply smoothing
        if self.prev_gaze_point is not None:
            gaze_point_2d = (
                GAZE_SMOOTHING_FACTOR * self.prev_gaze_point +
                (1 - GAZE_SMOOTHING_FACTOR) * gaze_point_2d
            )
        
        self.prev_gaze_point = gaze_point_2d
        gaze_data.gaze_point_2d = gaze_point_2d.astype(np.float32)
        
        # Normalized gaze point (0-1)
        gaze_data.gaze_point_3d = np.array([
            gaze_point_2d[0] / w,
            gaze_point_2d[1] / h,
            0.0
        ])
        
        # Estimate head pose using facial landmarks
        head_pose = self._estimate_head_pose(landmarks_3d)
        gaze_data.head_pose = head_pose
        
        return gaze_data
    
    @staticmethod
    def _compute_eye_aspect_ratio(landmarks_3d: np.ndarray, eye_indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink detection.
        EAR = ||p2 - p6|| + ||p3 - p5|| / (2 * ||p1 - p4||)
        
        Args:
            landmarks_3d: 3D facial landmarks [N, 3]
            eye_indices: Indices of eye landmarks
            
        Returns:
            Eye aspect ratio (float). Low values indicate blink.
        """
        if len(eye_indices) != 6:
            return 1.0
        
        pts = landmarks_3d[eye_indices]
        
        # Compute distances
        vertical_dist_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_dist_2 = np.linalg.norm(pts[2] - pts[4])
        horizontal_dist = np.linalg.norm(pts[0] - pts[3])
        
        ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist + 1e-6)
        return float(ear)
    
    @staticmethod
    def _compute_gaze_direction(
        eye_center: np.ndarray,
        iris_center: np.ndarray
    ) -> np.ndarray:
        """
        Compute gaze direction from iris position relative to eye center.
        
        Args:
            eye_center: 3D position of eye center
            iris_center: 3D position of iris/pupil center
            
        Returns:
            3D gaze direction vector (not normalized)
        """
        # The gaze direction is approximately toward the iris
        gaze_direction = iris_center - eye_center
        # Add a forward component for 3D direction
        gaze_direction = np.array([gaze_direction[0], gaze_direction[1], -0.3])
        return gaze_direction
    
    @staticmethod
    def _estimate_2d_gaze_point(
        left_eye_center: np.ndarray,
        right_eye_center: np.ndarray,
        left_iris: np.ndarray,
        right_iris: np.ndarray,
        landmarks_3d: np.ndarray,
        h: int,
        w: int
    ) -> np.ndarray:
        """
        Estimate 2D gaze point on screen using iris and facial landmarks.
        Uses a simple cross-ratio method.
        
        Args:
            left_eye_center: Left eye center (normalized)
            right_eye_center: Right eye center (normalized)
            left_iris: Left iris center (normalized)
            right_iris: Right iris center (normalized)
            landmarks_3d: All facial landmarks
            h, w: Frame height and width
            
        Returns:
            2D gaze point in pixel coordinates
        """
        # Compute iris offset from eye center (as normalized shift)
        left_offset = left_iris[:2] - left_eye_center[:2]
        right_offset = right_iris[:2] - right_eye_center[:2]
        
        # Average offset
        avg_offset = (left_offset + right_offset) / 2.0
        
        # Estimate gaze point from average eye position + scaled offset
        avg_eye = (left_eye_center[:2] + right_eye_center[:2]) / 2.0
        
        # Scale offset to screen coordinates
        scale_factor = 3.0  # Empirically determined scaling
        gaze_point = avg_eye + avg_offset * scale_factor
        
        # Clamp to frame bounds
        gaze_point[0] = np.clip(gaze_point[0] * w, 0, w - 1)
        gaze_point[1] = np.clip(gaze_point[1] * h, 0, h - 1)
        
        return gaze_point
    
    @staticmethod
    def _estimate_head_pose(landmarks_3d: np.ndarray) -> np.ndarray:
        """
        Estimate head pose (pitch, yaw, roll) from 3D facial landmarks.
        Uses a simple method based on landmark positions.
        
        Args:
            landmarks_3d: 3D facial landmarks [N, 3]
            
        Returns:
            Head pose as [pitch, yaw, roll] in degrees
        """
        # Key points for head pose estimation
        nose_tip = landmarks_3d[1]  # Nose tip
        left_eye = landmarks_3d[159]
        right_eye = landmarks_3d[386]
        left_mouth = landmarks_3d[61]
        right_mouth = landmarks_3d[291]
        
        # Compute head axes
        horizontal_axis = right_eye - left_eye
        vertical_axis = left_mouth - left_eye
        
        # Estimate angles
        yaw = np.arctan2(horizontal_axis[0], horizontal_axis[2]) * 180 / np.pi
        pitch = np.arctan2(vertical_axis[1], vertical_axis[2]) * 180 / np.pi
        
        # Simple roll estimation from eye level
        roll = 0.0  # Simplified
        
        return np.array([pitch, yaw, roll])
    
    def get_landmarks_for_visualization(self, gaze_data: GazeData) -> Optional[np.ndarray]:
        """Get landmarks in normalized format for visualization."""
        if gaze_data.landmarks_3d is None:
            return None
        return gaze_data.landmarks_3d[:, :2]  # Return only x, y coordinates


class GazeVisualizer:
    """Utilities for rendering gaze tracking data on video frames."""
    
    @staticmethod
    def draw_gaze_vector(
        frame: np.ndarray,
        gaze_data: GazeData,
        line_length: int = 100,
        thickness: int = 2,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw gaze vector line on frame."""
        if not gaze_data.face_detected:
            return frame
        
        h, w = frame.shape[:2]
        eye_center = (
            gaze_data.left_eye_center + gaze_data.right_eye_center
        ) / 2.0
        eye_center_px = (int(eye_center[0] * w), int(eye_center[1] * h))
        
        # Project gaze vector
        gaze_end = eye_center_px + (
            gaze_data.gaze_vector[:2] * line_length
        ).astype(int)
        
        cv2.line(frame, eye_center_px, tuple(gaze_end), color, thickness)
        cv2.circle(frame, eye_center_px, 5, color, -1)
        
        return frame
    
    @staticmethod
    def draw_pupils(
        frame: np.ndarray,
        gaze_data: GazeData,
        radius: int = 3,
        color: Tuple[int, int, int] = (0, 165, 255)
    ) -> np.ndarray:
        """Draw pupil positions on frame."""
        if not gaze_data.face_detected:
            return frame
        
        h, w = frame.shape[:2]
        
        # Left pupil
        left_pupil_px = (int(gaze_data.pupil_left[0] * w), int(gaze_data.pupil_left[1] * h))
        cv2.circle(frame, left_pupil_px, radius, color, -1)
        
        # Right pupil
        right_pupil_px = (int(gaze_data.pupil_right[0] * w), int(gaze_data.pupil_right[1] * h))
        cv2.circle(frame, right_pupil_px, radius, color, -1)
        
        return frame
    
    @staticmethod
    def draw_blink_indicator(
        frame: np.ndarray,
        gaze_data: GazeData,
        text_pos: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """Draw blink indicator on frame."""
        if gaze_data.is_blinking:
            cv2.putText(
                frame, "BLINK", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
        return frame

