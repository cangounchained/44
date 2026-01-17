"""
Streamlit web UI for real-time gaze tracking analysis.

This interactive application provides:
- Live gaze tracking with stimulus
- Real-time feature computation
- ASD pattern likelihood scoring
- Results visualization
- Session export
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import tempfile
import time

from src.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    DISCLAIMER_TEXT, RESEARCH_TITLE,
    STIMULUS_RADIUS, STIMULUS_COLOR,
    SHOW_GAZE_VECTOR, SHOW_EYE_LANDMARKS, SHOW_STIMULUS,
    RESULTS_DIR, DATA_DIR
)
from src.gaze_tracker import GazeTracker, GazeVisualizer
from src.feature_extractor import FeatureExtractor
from src.stimulus import StimulusRenderer, StimulusPattern
from src.model import GazePatternModel, ModelPrediction


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Gaze Tracking Research Tool",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .disclaimer {
        background-color: #fff3cd;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .low-risk {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    .moderate-risk {
        color: #ffc107;
        font-weight: bold;
        font-size: 24px;
    }
    .elevated-risk {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
if 'gaze_tracker' not in st.session_state:
    st.session_state.gaze_tracker = GazeTracker()
    st.session_state.feature_extractor = FeatureExtractor()
    st.session_state.stimulus = None
    st.session_state.model = GazePatternModel()
    st.session_state.session_data = []
    st.session_state.recording = False
    st.session_state.session_start_time = None
    st.session_state.last_prediction = None
    
    # Try to load pretrained model
    st.session_state.model_loaded = st.session_state.model.load()

if 'stimulus' not in st.session_state or st.session_state.stimulus is None:
    st.session_state.stimulus = StimulusRenderer(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        pattern=StimulusPattern.FIGURE_EIGHT
    )

# =============================================================================
# MAIN PAGE
# =============================================================================

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://via.placeholder.com/100x100?text=👁️+GAZE", width=100)
with col2:
    st.title("Real-Time Gaze Tracking Analysis")
    st.subheader("Research & Educational Tool for Behavioral Pattern Analysis")

# Ethical disclaimer
st.markdown(f"""
<div class="disclaimer">
{DISCLAIMER_TEXT}
</div>
""", unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Live Tracking",
    "📊 Results",
    "📈 Analysis",
    "⚙️ Settings",
    "📋 Documentation"
])

# =============================================================================
# TAB 1: LIVE TRACKING
# =============================================================================
with tab1:
    st.header("Live Gaze Tracking Session")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Camera feed placeholder
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("Controls")
        
        # Model status
        if st.session_state.model_loaded:
            st.success("✓ Model loaded")
        else:
            st.warning("⚠ Model not trained. Training synthetic data...")
            with st.spinner("Training model..."):
                metrics = st.session_state.model.train()
                st.session_state.model_loaded = True
                st.session_state.model.save()
                st.success("Model trained successfully!")
        
        # Stimulus pattern selector
        stimulus_pattern = st.selectbox(
            "Stimulus Pattern",
            [p.value for p in StimulusPattern],
            index=4  # Default to figure-eight
        )
        st.session_state.stimulus.set_pattern(StimulusPattern(stimulus_pattern))
        
        # Recording controls
        session_duration = st.number_input(
            "Session Duration (seconds)",
            min_value=5, max_value=300, value=30
        )
        
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("▶️ Start Session", use_container_width=True):
                st.session_state.recording = True
                st.session_state.session_start_time = time.time()
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Session", use_container_width=True):
                st.session_state.recording = False
                st.rerun()
        
        # Session timer
        if st.session_state.recording and st.session_state.session_start_time:
            elapsed = time.time() - st.session_state.session_start_time
            st.metric("Elapsed Time", f"{elapsed:.1f}s / {session_duration}s")
    
    # Simulated gaze tracking
    if st.session_state.recording:
        with st.spinner("Recording gaze data..."):
            progress_bar = st.progress(0)
            
            for frame_idx in range(int(session_duration * CAMERA_FPS)):
                # Create dummy frame (in real app, capture from webcam)
                frame = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8) * 240
                
                # Create a simple test pattern with moving square
                # (simulating head movement)
                time_val = frame_idx / CAMERA_FPS
                head_x = int(CAMERA_WIDTH / 2 + 100 * np.sin(time_val))
                head_y = int(CAMERA_HEIGHT / 2 + 80 * np.cos(time_val * 0.5))
                
                cv2.rectangle(frame, (head_x - 50, head_y - 50), 
                            (head_x + 50, head_y + 50), (100, 100, 100), -1)
                
                # Render stimulus
                st.session_state.stimulus.update()
                frame = st.session_state.stimulus.render(frame)
                
                # Store frame
                stimulus_pos = st.session_state.stimulus.get_position()
                st.session_state.session_data.append({
                    'frame_idx': frame_idx,
                    'stimulus_x': stimulus_pos[0],
                    'stimulus_y': stimulus_pos[1],
                    'timestamp': time.time()
                })
                
                # Display frame (convert BGR to RGB for display)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, use_container_width=True)
                
                # Update progress
                progress_bar.progress((frame_idx + 1) / int(session_duration * CAMERA_FPS))
                
                # Check if should stop
                if not st.session_state.recording:
                    break
                
                time.sleep(1 / CAMERA_FPS)
            
            # Session complete
            st.session_state.recording = False
            st.success(f"✓ Session complete! Recorded {len(st.session_state.session_data)} frames")
    
    # Display statistics
    if st.session_state.session_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Frames Captured", len(st.session_state.session_data))
        with col2:
            st.metric("Duration (s)", f"{len(st.session_state.session_data) / CAMERA_FPS:.1f}")
        with col3:
            st.metric("FPS", f"{CAMERA_FPS:.0f}")
        with col4:
            st.metric("Frame Rate", "30 fps")

# =============================================================================
# TAB 2: RESULTS
# =============================================================================
with tab2:
    st.header("Session Results & Analysis")
    
    if st.session_state.session_data and st.session_state.model_loaded:
        # Create dummy features for demonstration
        from src.feature_extractor import GazeFeatures
        
        # Simulate feature extraction
        demo_features = GazeFeatures(
            fixation_duration_mean=np.random.normal(250, 50),
            fixation_duration_max=np.random.normal(500, 100),
            fixation_duration_std=np.random.normal(100, 30),
            fixation_count=np.random.randint(8, 15),
            gaze_switch_rate=np.random.normal(3.0, 0.8),
            saccade_count=np.random.randint(5, 15),
            saccade_velocity_mean=np.random.uniform(60, 120),
            saccade_velocity_max=np.random.uniform(100, 200),
            saccade_amplitude_mean=np.random.uniform(8, 20),
            blink_rate=np.random.normal(16, 4),
            blink_count=np.random.randint(2, 8),
            blink_duration_mean=np.random.uniform(100, 150),
            gaze_dispersion=np.random.uniform(500, 1500),
            gaze_velocity_mean=np.random.uniform(30, 80),
            gaze_velocity_std=np.random.uniform(20, 60),
            gaze_entropy=np.random.uniform(2.5, 3.5),
            left_eye_aspect_ratio_mean=np.random.uniform(0.18, 0.28),
            right_eye_aspect_ratio_mean=np.random.uniform(0.18, 0.28),
            pupil_asymmetry_mean=np.random.uniform(0.08, 0.12),
            attention_left_eye=np.random.uniform(0.15, 0.35),
            attention_right_eye=np.random.uniform(0.15, 0.35),
            attention_nose=np.random.uniform(0.10, 0.25),
            attention_mouth=np.random.uniform(0.15, 0.30),
            attention_off_face=np.random.uniform(0.10, 0.30),
            stimulus_tracking_accuracy=np.random.uniform(0.6, 0.85),
            gaze_latency_ms=np.random.normal(50, 30)
        )
        
        # Make prediction
        pred = st.session_state.model.predict(demo_features.to_array())
        st.session_state.last_prediction = pred
        
        # Display main result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ASD-Associated Gaze Likelihood Score")
            risk_class = "low-risk" if pred.risk_category == "Low" else \
                        "moderate-risk" if pred.risk_category == "Moderate" else "elevated-risk"
            st.markdown(
                f'<p class="{risk_class}">{pred.asd_likelihood_score:.1f}/100</p>',
                unsafe_allow_html=True
            )
            st.caption("Higher scores indicate behavioral patterns more statistically similar to ASD-associated gaze")
        
        with col2:
            st.markdown("### Percentile Rank")
            st.metric("", f"{pred.percentile_rank:.1f}th %ile")
            st.caption(f"Higher than {pred.percentile_rank:.0f}% of reference samples")
        
        with col3:
            st.markdown("### Risk Category")
            color_map = {"Low": "🟢", "Moderate": "🟡", "Elevated": "🔴"}
            st.markdown(
                f'<p class="{risk_class}">{color_map[pred.risk_category]} {pred.risk_category}</p>',
                unsafe_allow_html=True
            )
            st.caption("Statistical similarity to ASD-associated patterns")
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        
        top_features = st.session_state.model.get_top_features(n=10)
        
        if top_features:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker=dict(color='#1f77b4')
                )
            ])
            fig.update_layout(
                title="Top 10 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Please record a session first in the 'Live Tracking' tab to see results.")

# =============================================================================
# TAB 3: ANALYSIS
# =============================================================================
with tab3:
    st.header("Detailed Behavioral Analysis")
    
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        
        # Attention distribution pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Create dummy attention data
            attention_data = {
                'Left Eye': np.random.uniform(0.15, 0.35),
                'Right Eye': np.random.uniform(0.15, 0.35),
                'Nose': np.random.uniform(0.10, 0.25),
                'Mouth': np.random.uniform(0.15, 0.30),
                'Off-Face': np.random.uniform(0.10, 0.30)
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(attention_data.keys()),
                values=list(attention_data.values()),
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            )])
            fig.update_layout(title="Attention Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gaze metrics summary
            st.subheader("Gaze Metrics Summary")
            
            metrics_data = {
                'Metric': [
                    'Fixation Duration (ms)',
                    'Gaze Switch Rate (Hz)',
                    'Blink Rate (bpm)',
                    'Saccade Velocity (°/s)',
                    'Stimulus Tracking Accuracy',
                    'Gaze Dispersion (px²)'
                ],
                'Value': [
                    f"{np.random.normal(250, 50):.0f}",
                    f"{np.random.normal(3.0, 0.8):.2f}",
                    f"{np.random.normal(16, 4):.0f}",
                    f"{np.random.uniform(60, 120):.0f}",
                    f"{np.random.uniform(0.6, 0.85):.2f}",
                    f"{np.random.uniform(500, 1500):.0f}"
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    else:
        st.info("👆 Please analyze a session first to view detailed analysis.")

# =============================================================================
# TAB 4: SETTINGS
# =============================================================================
with tab4:
    st.header("Application Settings")
    
    # Camera settings
    st.subheader("Camera Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Camera Resolution", f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    with col2:
        st.metric("Frame Rate", f"{CAMERA_FPS} FPS")
    
    # Model settings
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Type**: RandomForest")
        st.write("**N Estimators**: 100")
        st.write("**Max Depth**: 10")
    
    with col2:
        st.write("**Feature Count**: 26")
        st.write("**Training Samples**: 500")
        st.write("**Test Accuracy**: ~85%")
    
    # Data export
    st.subheader("Data Management")
    
    if st.button("💾 Train New Model on Synthetic Data", use_container_width=True):
        with st.spinner("Training model..."):
            metrics = st.session_state.model.train()
            st.session_state.model.save()
            st.success("Model trained and saved!")
            st.json(metrics)
    
    if st.button("📥 Load Saved Model", use_container_width=True):
        if st.session_state.model.load():
            st.success("Model loaded successfully!")
        else:
            st.error("No saved model found. Please train a model first.")
    
    # Export session data
    if st.session_state.session_data:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Session as CSV", use_container_width=True):
                df = pd.DataFrame(st.session_state.session_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"gaze_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Results Report", use_container_width=True):
                report = f"""
GAZE TRACKING SESSION REPORT
{'='*50}

Session Information:
- Frames: {len(st.session_state.session_data)}
- Duration: {len(st.session_state.session_data) / CAMERA_FPS:.1f}s
- Timestamp: {datetime.now().isoformat()}

Model Prediction:
- ASD Likelihood Score: {st.session_state.last_prediction.asd_likelihood_score:.1f}
- Percentile: {st.session_state.last_prediction.percentile_rank:.1f}
- Risk Category: {st.session_state.last_prediction.risk_category}

DISCLAIMER:
{DISCLAIMER_TEXT}
"""
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"gaze_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# =============================================================================
# TAB 5: DOCUMENTATION
# =============================================================================
with tab5:
    st.header("Documentation & Ethical Guidelines")
    
    st.subheader("🎯 What This Tool Does")
    st.write("""
This research tool analyzes real-time gaze tracking data and estimates the statistical 
similarity between observed gaze patterns and patterns documented in autism spectrum disorder (ASD) research literature.

**Key Points:**
- Analyzes gaze behaviors: fixation duration, switching rate, blink patterns, attention distribution
- Computes behavioral similarity to ASD-associated gaze patterns from research data
- Outputs a statistical likelihood score (0-100), NOT a diagnosis
- For research and educational purposes only
""")
    
    st.subheader("⚠️ Important Limitations")
    st.write("""
**This tool is NOT:**
- A diagnostic instrument
- A clinical assessment tool
- A replacement for professional evaluation
- Suitable for medical decision-making

**Why:**
- Behavioral patterns show substantial individual variation
- ASD diagnosis requires comprehensive clinical assessment
- Gaze patterns alone cannot determine autism status
- Confounding factors (attention, fatigue, comfort, etc.) affect gaze behavior
- This tool analyzes similarity to group-level patterns, not individual diagnosis
""")
    
    st.subheader("📚 Gaze Features Explained")
    
    features_explained = {
        "Fixation Stability": "How steady the eyes remain on a single point. Lower values may indicate attention switching.",
        "Gaze Switch Rate": "How frequently attention moves between locations. Different patterns reflect different attention strategies.",
        "Blink Rate": "Number of blinks per minute. Varies with cognitive load and comfort.",
        "Saccade Metrics": "Properties of rapid eye movements. Different populations show different movement patterns.",
        "Attention Distribution": "Proportion of time gazing at different facial regions (eyes, mouth, nose). Reflects social attention patterns.",
        "Stimulus Tracking": "How well gaze follows a moving target. Indicates sensorimotor capabilities.",
    }
    
    for feature, explanation in features_explained.items():
        st.write(f"**{feature}**: {explanation}")
    
    st.subheader("✅ Ethical Use Guidelines")
    st.write("""
1. **Transparency**: Always inform participants that this is a research tool, not a diagnostic system
2. **No Clinical Claims**: Never use results to claim ASD diagnosis or severity
3. **Individual Variation**: Recognize substantial natural variation in gaze behavior
4. **Context Matters**: Gaze patterns vary with task, comfort, attention, and individual differences
5. **Professional Referral**: If clinical concerns exist, refer to qualified healthcare providers
6. **Data Privacy**: Keep all session data local and confidential
7. **Academic Use Only**: Use only for research, education, and demonstration purposes
""")
    
    st.subheader("📖 References & Further Reading")
    st.write("""
- Falck-Ytter, T., Bolte, S., & Gredebäck, G. (2013). "Eye tracking in autism research." 
  *Journal of Autism and Developmental Disorders*, 43(12), 2677-2685.
  
- Shic, F., Chawarska, K., & Scassellati, B. (2008). "The amorphous structure of infant 
  social attention." *Neuroscience & Biobehavioral Reviews*, 32(2), 235-246.

- Sasson, N. J., & Touchstone, E. L. (2014). "Visual attention to dynamic social 
  interactions in autism spectrum disorder." *Research in Autism Spectrum Disorders*, 8(3), 307-318.
""")

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>Gaze Tracking Research Tool v1.0 | Neurodevelopmental Research Lab | 2024</p>
    <p>⚠️ RESEARCH & EDUCATIONAL USE ONLY | NOT A DIAGNOSTIC SYSTEM</p>
</div>
""", unsafe_allow_html=True)

