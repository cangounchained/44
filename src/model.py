"""
Machine learning models for ASD-associated gaze pattern classification.

Provides:
- Synthetic training data generation
- Model training and evaluation
- Inference with confidence scores
- Feature importance computation
- Percentile ranking
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.config import (
    MODELS_DIR, MODEL_TYPE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_RANDOM_STATE,
    N_SYNTHETIC_SAMPLES, N_ASD_POSITIVE, N_ASD_NEGATIVE,
    SYNTHETIC_PARAMS_HEALTHY, SYNTHETIC_PARAMS_ASD
)
from src.feature_extractor import GazeFeatures


@dataclass
class ModelPrediction:
    """Container for model prediction results."""
    asd_likelihood_score: float  # 0-100
    confidence: float  # 0-1
    percentile_rank: float  # 0-100
    risk_category: str  # "Low", "Moderate", "Elevated"
    feature_importance: Dict[str, float]
    decision_boundary: float


class SyntheticDataGenerator:
    """Generate synthetic training data with controlled characteristics."""
    
    @staticmethod
    def generate_dataset(
        n_asd_positive: int = N_ASD_POSITIVE,
        n_asd_negative: int = N_ASD_NEGATIVE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic gaze features for ASD and healthy controls.
        
        Args:
            n_asd_positive: Number of ASD-associated patterns
            n_asd_negative: Number of healthy control patterns
            
        Returns:
            (features, labels): Feature matrix and binary labels
        """
        features_asd = []
        features_healthy = []
        
        # Generate ASD-associated patterns
        for _ in range(n_asd_positive):
            features_asd.append(
                SyntheticDataGenerator._generate_sample(SYNTHETIC_PARAMS_ASD)
            )
        
        # Generate healthy control patterns
        for _ in range(n_asd_negative):
            features_healthy.append(
                SyntheticDataGenerator._generate_sample(SYNTHETIC_PARAMS_HEALTHY)
            )
        
        # Combine and create labels
        X = np.vstack([features_asd, features_healthy])
        y = np.hstack([
            np.ones(n_asd_positive),
            np.zeros(n_asd_negative)
        ])
        
        return X, y
    
    @staticmethod
    def _generate_sample(params: Dict) -> np.ndarray:
        """Generate a single synthetic sample from parameter distribution."""
        # Fixation metrics
        fix_duration_mean = np.random.normal(
            params['fixation_duration_mean'],
            params['fixation_duration_std']
        )
        fix_duration_max = fix_duration_mean * np.random.uniform(1.5, 3.0)
        fix_duration_std = fix_duration_mean * 0.3
        fix_count = int(np.random.normal(10, 3))
        
        # Gaze switching
        gaze_switch_rate = np.random.normal(
            params['gaze_switch_rate'],
            params['gaze_switch_std']
        )
        saccade_count = int(max(0, np.random.normal(gaze_switch_rate * 3, 2)))
        saccade_velocity_mean = np.random.uniform(50, 150)
        saccade_velocity_max = saccade_velocity_mean * 1.5
        saccade_amplitude_mean = np.random.uniform(5, 20)
        
        # Blink metrics
        blink_rate = np.random.normal(params['blink_rate'], params['blink_rate_std'])
        blink_count = int(max(0, blink_rate / 12))  # Per 2 second window
        blink_duration_mean = np.random.uniform(100, 200)
        
        # Gaze stability
        gaze_dispersion = np.random.uniform(100, 2000)
        gaze_velocity_mean = np.random.uniform(20, 100)
        gaze_velocity_std = gaze_velocity_mean * 0.5
        gaze_entropy = np.random.uniform(2, 4)
        
        # Eye metrics
        ear_left = np.random.uniform(0.15, 0.35)
        ear_right = np.random.uniform(0.15, 0.35)
        pupil_asymmetry = np.random.uniform(0.05, 0.15)
        
        # Attention distribution (must sum to 1)
        attention_left_eye = params['eye_attention'] * np.random.uniform(0.8, 1.2)
        attention_right_eye = params['eye_attention'] * np.random.uniform(0.8, 1.2)
        attention_nose = params['nose_attention'] * np.random.uniform(0.7, 1.3)
        attention_mouth = params['mouth_attention'] * np.random.uniform(0.7, 1.3)
        attention_off_face = params['off_face_attention'] * np.random.uniform(0.7, 1.3)
        
        # Normalize attention to sum to 1
        total_attention = (attention_left_eye + attention_right_eye + 
                         attention_nose + attention_mouth + attention_off_face)
        attention_left_eye /= total_attention
        attention_right_eye /= total_attention
        attention_nose /= total_attention
        attention_mouth /= total_attention
        attention_off_face /= total_attention
        
        # Stimulus tracking
        stimulus_tracking = params['tracking_accuracy']
        stimulus_latency = np.random.uniform(0, 200)
        
        # Create feature vector
        features = np.array([
            max(0, fix_duration_mean),
            max(0, fix_duration_max),
            max(0, fix_duration_std),
            max(0, fix_count),
            max(0, gaze_switch_rate),
            max(0, saccade_count),
            max(0, saccade_velocity_mean),
            max(0, saccade_velocity_max),
            max(0, saccade_amplitude_mean),
            max(0, blink_rate),
            max(0, blink_count),
            max(0, blink_duration_mean),
            max(0, gaze_dispersion),
            max(0, gaze_velocity_mean),
            max(0, gaze_velocity_std),
            max(0, gaze_entropy),
            ear_left,
            ear_right,
            pupil_asymmetry,
            attention_left_eye,
            attention_right_eye,
            attention_nose,
            attention_mouth,
            attention_off_face,
            stimulus_tracking,
            stimulus_latency
        ])
        
        return features


class GazePatternModel:
    """
    ML model for ASD-associated gaze pattern classification.
    
    Attributes:
        - Uses RandomForest for interpretability
        - Computes feature importance
        - Supports probability calibration
    """
    
    def __init__(self):
        """Initialize model components."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = GazeFeatures.get_feature_names()
        self.is_fitted = False
        self.training_scores = None  # For percentile ranking
        self.decision_boundary = 0.5
    
    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train model on feature data.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Binary labels [n_samples]
            test_size: Proportion for test set
            
        Returns:
            Dictionary of training metrics
        """
        # Generate synthetic data if not provided
        if X is None or y is None:
            print("Generating synthetic training data...")
            X, y = SyntheticDataGenerator.generate_dataset()
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=RF_RANDOM_STATE,
            stratify=y
        )
        
        # Train RandomForest
        print("Training RandomForest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store training scores for percentile computation
        self.training_scores = self.model.predict_proba(X_scaled)[:, 1]
        
        self.is_fitted = True
        
        metrics = {
            'accuracy': accuracy,
            'auc': auc,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"Training complete: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
        return metrics
    
    def predict(self, features: np.ndarray) -> ModelPrediction:
        """
        Predict ASD-associated gaze pattern likelihood.
        
        Args:
            features: Feature vector or array [n_features] or [n_samples, n_features]
            
        Returns:
            ModelPrediction with score and explanation
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize
        features_scaled = self.scaler.transform(features)
        
        # Get prediction
        pred_proba = self.model.predict_proba(features_scaled)[0, 1]
        
        # Convert to 0-100 scale
        asd_score = float(pred_proba * 100)
        
        # Compute percentile
        if self.training_scores is not None:
            percentile = (
                np.sum(self.training_scores <= pred_proba) /
                len(self.training_scores) * 100
            )
        else:
            percentile = pred_proba * 100
        
        # Determine risk category
        if percentile < 33:
            risk_category = "Low"
        elif percentile < 67:
            risk_category = "Moderate"
        else:
            risk_category = "Elevated"
        
        # Get feature importance
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        return ModelPrediction(
            asd_likelihood_score=asd_score,
            confidence=float(abs(pred_proba - 0.5) * 2),  # Confidence based on distance from boundary
            percentile_rank=float(percentile),
            risk_category=risk_category,
            feature_importance=feature_importance,
            decision_boundary=float(self.decision_boundary)
        )
    
    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """
        Get top n most important features.
        
        Args:
            n: Number of features to return
            
        Returns:
            Dictionary of feature names and importance scores
        """
        if not self.is_fitted:
            return {}
        
        importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort and return top n
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:n])
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (default: models/gaze_model.pkl)
            
        Returns:
            Path to saved model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filepath is None:
            filepath = MODELS_DIR / "gaze_model.pkl"
        
        filepath.parent.mkdir(exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_scores': self.training_scores,
            'decision_boundary': self.decision_boundary
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: Path = MODELS_DIR / "gaze_model.pkl") -> bool:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successfully loaded
        """
        if not filepath.exists():
            print(f"Model file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_scores = model_data.get('training_scores')
        self.decision_boundary = model_data.get('decision_boundary', 0.5)
        self.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        return True

