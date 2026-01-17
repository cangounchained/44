"""
Stimulus rendering module for gaze tracking experiments.

Renders moving red dot stimulus with predictable and semi-random patterns.
"""

import numpy as np
import cv2
from enum import Enum
from typing import Tuple
from src.config import (
    STIMULUS_RADIUS, STIMULUS_COLOR, STIMULUS_SPEED,
    STIMULUS_CHANGE_INTERVAL, STIMULUS_DWELL_TIME,
    CAMERA_WIDTH, CAMERA_HEIGHT
)


class StimulusPattern(Enum):
    """Types of stimulus movement patterns."""
    LINEAR = "linear"
    CIRCULAR = "circular"
    RANDOM = "random"
    GRID = "grid"
    FIGURE_EIGHT = "figure_eight"


class StimulusRenderer:
    """
    Render dynamic stimulus for gaze tracking experiments.
    
    Supports multiple movement patterns:
    - Linear: Smooth movement in straight lines
    - Circular: Circular motion around center
    - Random: Semi-random smooth paths
    - Grid: Movement to grid points
    - Figure-eight: Lemniscate path
    """
    
    def __init__(
        self,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        radius: int = STIMULUS_RADIUS,
        color: Tuple[int, int, int] = STIMULUS_COLOR,
        speed: float = STIMULUS_SPEED,
        pattern: StimulusPattern = StimulusPattern.FIGURE_EIGHT
    ):
        """
        Initialize stimulus renderer.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            radius: Stimulus circle radius in pixels
            color: Color in BGR format
            speed: Movement speed in pixels per frame
            pattern: Type of movement pattern
        """
        self.width = width
        self.height = height
        self.radius = radius
        self.color = color
        self.speed = speed
        self.pattern = pattern
        
        # Current state
        self.position = np.array([width / 2.0, height / 2.0])
        self.velocity = np.array([speed, 0.0])
        self.frame_count = 0
        self.pattern_change_frame = 0
        
        # Pattern-specific state
        self.pattern_state = 0
        self.grid_index = 0
        self.grid_points = self._generate_grid_points()
        
        # Smooth target for random walks
        self.target_position = self.position.copy()
        self.target_change_frame = STIMULUS_CHANGE_INTERVAL
    
    def update(self) -> np.ndarray:
        """
        Update stimulus position.
        
        Returns:
            Current position as [x, y] in pixels
        """
        if self.pattern == StimulusPattern.LINEAR:
            self._update_linear()
        elif self.pattern == StimulusPattern.CIRCULAR:
            self._update_circular()
        elif self.pattern == StimulusPattern.RANDOM:
            self._update_random()
        elif self.pattern == StimulusPattern.GRID:
            self._update_grid()
        elif self.pattern == StimulusPattern.FIGURE_EIGHT:
            self._update_figure_eight()
        
        # Clamp to bounds
        self.position[0] = np.clip(self.position[0], self.radius, self.width - self.radius)
        self.position[1] = np.clip(self.position[1], self.radius, self.height - self.radius)
        
        self.frame_count += 1
        return self.position.copy()
    
    def _update_linear(self) -> None:
        """Update for linear movement pattern."""
        # Move in current direction
        self.position += self.velocity
        
        # Bounce off walls
        if self.position[0] - self.radius <= 0 or self.position[0] + self.radius >= self.width:
            self.velocity[0] *= -1
        if self.position[1] - self.radius <= 0 or self.position[1] + self.radius >= self.height:
            self.velocity[1] *= -1
    
    def _update_circular(self) -> None:
        """Update for circular movement pattern."""
        # Circular motion around center
        center = np.array([self.width / 2.0, self.height / 2.0])
        radius = min(self.width, self.height) / 3.0
        
        angle = self.frame_count * self.speed / (radius + 1e-6)
        self.position[0] = center[0] + radius * np.cos(angle)
        self.position[1] = center[1] + radius * np.sin(angle)
    
    def _update_random(self) -> None:
        """Update for random smooth movement pattern."""
        # Move toward random target with smooth transitions
        direction = self.target_position - self.position
        dist = np.linalg.norm(direction)
        
        if dist > 1:
            direction = direction / dist
            self.position += direction * self.speed
        
        # Change target periodically
        if self.frame_count >= self.target_change_frame:
            self.target_position = np.array([
                np.random.uniform(self.radius, self.width - self.radius),
                np.random.uniform(self.radius, self.height - self.radius)
            ])
            self.target_change_frame = self.frame_count + STIMULUS_CHANGE_INTERVAL
    
    def _update_grid(self) -> None:
        """Update for grid-based movement pattern."""
        # Dwell at grid points before moving to next
        current_target = self.grid_points[self.grid_index]
        direction = current_target - self.position
        dist = np.linalg.norm(direction)
        
        if dist > self.speed:
            direction = direction / (dist + 1e-6)
            self.position += direction * self.speed
        else:
            # Reached target, move to next grid point
            self.position = current_target.copy()
            self.grid_index = (self.grid_index + 1) % len(self.grid_points)
    
    def _update_figure_eight(self) -> None:
        """Update for figure-eight (lemniscate) movement pattern."""
        # Lemniscate equation: r^2 = a^2 * cos(2*theta)
        center = np.array([self.width / 2.0, self.height / 2.0])
        a = min(self.width, self.height) / 4.0
        
        t = self.frame_count * 0.02  # Parameter for smooth motion
        
        # Lemniscate parametric equations
        denom = np.cos(2 * t) + 1 + 1e-6
        x = a * np.cos(t) / denom
        y = a * np.sin(2 * t) / (2 * denom)
        
        self.position[0] = center[0] + x
        self.position[1] = center[1] + y
    
    def _generate_grid_points(self, rows: int = 3, cols: int = 3) -> np.ndarray:
        """Generate grid points for grid pattern."""
        points = []
        margin = self.radius + 50
        
        for i in range(rows):
            for j in range(cols):
                x = margin + (self.width - 2 * margin) * j / max(cols - 1, 1)
                y = margin + (self.height - 2 * margin) * i / max(rows - 1, 1)
                points.append([x, y])
        
        return np.array(points)
    
    def render(self, frame: np.ndarray) -> np.ndarray:
        """
        Render stimulus on frame.
        
        Args:
            frame: Input frame (H, W, 3)
            
        Returns:
            Frame with stimulus rendered
        """
        if frame is None:
            return frame
        
        # Convert position to integers
        pos_int = self.position.astype(int)
        
        # Draw circle
        cv2.circle(frame, tuple(pos_int), self.radius, self.color, -1)
        
        # Draw crosshair
        cross_size = self.radius + 5
        cv2.line(
            frame,
            (pos_int[0] - cross_size, pos_int[1]),
            (pos_int[0] + cross_size, pos_int[1]),
            (255, 255, 255), 1
        )
        cv2.line(
            frame,
            (pos_int[0], pos_int[1] - cross_size),
            (pos_int[0], pos_int[1] + cross_size),
            (255, 255, 255), 1
        )
        
        return frame
    
    def get_position(self) -> Tuple[float, float]:
        """Get current stimulus position."""
        return tuple(self.position)
    
    def set_pattern(self, pattern: StimulusPattern) -> None:
        """Change stimulus movement pattern."""
        self.pattern = pattern
        self.pattern_state = 0
    
    def reset(self) -> None:
        """Reset stimulus to initial state."""
        self.position = np.array([self.width / 2.0, self.height / 2.0])
        self.velocity = np.array([self.speed, 0.0])
        self.frame_count = 0
        self.pattern_state = 0
        self.grid_index = 0
        self.target_position = self.position.copy()
        self.target_change_frame = STIMULUS_CHANGE_INTERVAL

