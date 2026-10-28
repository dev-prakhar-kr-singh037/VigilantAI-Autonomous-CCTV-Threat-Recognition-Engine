import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom UI Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CCTV Violence & Weapon Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Model Loading
# -----------------------------------------------------------------------------
MODEL_PATH = "weights/best.pt"  # Place best.pt in the same folder as app.py

@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file '{path}' not found in the application directory!")
        st.stop()
    return YOLO(path)

with st.spinner("Initializing YOLOv8 Model & Environment..."):
    model = load_yolo_model(MODEL_PATH)

# Class name mapping
class_names = [model.names[i] for i in range(len(model.names))]

# -----------------------------------------------------------------------------
# 3. Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric/100/security-camera.png", width=70)
st.sidebar.title("System Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=30,
    max_value=95,
    value=60,
    step=5
) / 100.0

window_size = st.sidebar.slider(
    "Smoothing Window (Frames)",
    min_value=1,
    max_value=25,
    value=10,
    help="Averages predictions over continuous frames to prevent false flicker."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Target Categories")
for cls in class_names:
    st.sidebar.markdown(f"- **{cls}**")

# -----------------------------------------------------------------------------
# 4. Main Application Dashboard
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🚨 Smart City Violence & Weapon Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered automated CCTV surveillance monitoring system</div>', unsafe_allow_html=True)

uploaded_video = st.file_uploader("Upload CCTV Video Clip (.mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None:
    # Save uploaded video to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Surveillance Feed")
        st_frame = st.empty()

    with col2:
        st.subheader("⚠️ Real-Time Status")
        status_box = st.empty()
        conf_metric = st.empty()
        frame_counter = st.empty()

    # Video Processing Loop Setup
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prob_history = []
    detection_records = []
    frame_index = 0

    stop_button = st.button("⏹ Stop Processing")

    progress_bar = st.progress(0)

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # Run YOLO inference
        results = model.predict(frame, imgsz=416, verbose=False)[0]
        raw_probs = results.probs.data.cpu().numpy()  # Class probabilities

        # Add to sliding window history
        prob_history.append(raw_probs)
        if len(prob_history) > window_size:
            prob_history.pop(0)

        # Average probabilities across window
        smoothed_probs = np.mean(prob_history, axis=0)
        top_class_id = np.argmax(smoothed_probs)
        predicted_class = model.names[top_class_id]
        confidence = smoothed_probs[top_class_id]

        # Save record for analytics
        detection_records.append({
            "frame": frame_index,
            "class": predicted_class,
            "confidence": confidence,
            **{model.names[i]: float(smoothed_probs[i]) for i in range(len(model.names))}
        })

        # Define alert styling and overlays based on predicted state
        cls_lower = predicted_class.lower()
        if cls_lower == 'normal':
            color_bgr = (0, 200, 0)      # Green
            status_html = "<h3 style='color:#16A34A;'>🟢 NORMAL SCENE</h3>"
        elif cls_lower == 'violence':
            color_bgr = (0, 140, 255)    # Orange
            status_html = "<h3 style='color:#EA580C;'>⚠️ VIOLENCE DETECTED</h3>"
        else:  # Weaponized
            color_bgr = (0, 0, 230)      # Red
            status_html = "<h3 style='color:#DC2626;'>🚨 WEAPON DETECTED!</h3>"

        # Apply overlay text directly on video frame
        label_text = f"{predicted_class.upper()}: {confidence:.1%}"
        cv2.rectangle(frame, (15, 15), (420, 70), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2, cv2.LINE_AA)

        # Display updated video frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update metrics box
        status_box.markdown(status_html, unsafe_allow_html=True)
        conf_metric.metric("Confidence Score", f"{confidence:.1%}")
        frame_counter.caption(f"Frame {frame_index} of {total_frames}")

        if total_frames > 0:
            progress_bar.progress(min(frame_index / total_frames, 1.0))

    cap.release()
    st.success("✅ Surveillance video processing complete!")

    # -----------------------------------------------------------------------------
    # 5. Post-Analysis Analytics & Charts
    # -----------------------------------------------------------------------------
    if detection_records:
        df_results = pd.DataFrame(detection_records)

        st.markdown("---")
        st.subheader("📈 Sequence Detection Analytics")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("**Class Distribution Across Video**")
            class_counts = df_results['class'].value_counts().reset_index()
            class_counts.columns = ['Category', 'Frame Count']
            fig_pie = px.pie(class_counts, values='Frame Count', names='Category', hole=0.4,
                             color='Category',
                             color_discrete_map={'Normal': '#22C55E', 'Violence': '#F97316', 'Weaponized': '#EF4444'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_chart2:
            st.markdown("**Confidence Trend Across Timeline**")
            fig_line = px.line(df_results, x='frame', y=[c for c in class_names if c in df_results.columns],
                               labels={'value': 'Probability', 'frame': 'Frame Index'},
                               title="Probability per Class over Time")
            st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("👆 Please upload a CCTV video file to begin analysis.")