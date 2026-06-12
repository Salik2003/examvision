import streamlit as st
import cv2
from ultralytics import YOLO
import utils.detection as detection
from utils.alert_system import AlertSystem, AlertNotifier, CHEATING_BEHAVIORS
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import os
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Create detections folder if it doesn't exist
os.makedirs("cheating_detections", exist_ok=True)

# Initialize Alert System
alert_system = AlertSystem()

# RTC configuration using self-hosted coturn TURN server
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:13.212.250.73:3478"],
            "username": "exam",
            "credential": "vision123",
        },
        {
            "urls": ["turn:13.212.250.73:3478?transport=tcp"],
            "username": "exam",
            "credential": "vision123",
        },
    ]
})

# Classes the model was trained with — loaded dynamically from model.names.
# These are fallback names only used if the model can't be loaded.
class_names = {
    0: 'cheating', 1: 'good', 2: 'normal'
}

# Known cheating class names across all model variants
CHEATING_CLASSES_FULL = {
    'cheating', 'look_left', 'look_right', 'look_up', 'look_down',
    'mouth_open', 'see_left', 'see_right', 'see_down'
}

def get_cheating_classes(model_names: dict) -> set:
    """Return the set of class names that represent cheating for a given model."""
    names = set(model_names.values())
    # If model has explicit 'cheating' class, use that plus any head-turn classes
    if 'cheating' in names:
        return names & CHEATING_CLASSES_FULL
    # Otherwise use all known cheating class names that appear in this model
    matched = names & CHEATING_CLASSES_FULL
    return matched if matched else names - {'good', 'normal', 'look_forward', 'see_forward', 'mouth_close', 'Face'}

# Model paths
model_paths = {
    "YOLOv8 Full (New)": 'yolov8/trained_model_50_epochs.pt',
    "YOLOv8 Standard": 'yolov8/trained_model.pt',
    "YOLOv8 OBB": 'yolov8/yolov8-obb_trained.pt'
}

# Load YOLO models
@st.cache_resource
def load_models():
    loaded = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                loaded[name] = YOLO(path)
            except:
                loaded[name] = YOLO('yolov8n.pt')
        else:
            loaded[name] = YOLO('yolov8n.pt')
    return loaded

models = load_models()

# Streamlit UI
st.set_page_config(page_title="Exam Monitoring System", layout="wide")
st.sidebar.title('🎓 Exam Monitoring System')

# Page selection
page = st.sidebar.radio("Select Page", ('Detection', 'Alert Dashboard', 'Comparison'))

# Selection between live video and file upload
source = st.sidebar.radio("Select video source", ('Live Video', 'Upload MP4 File'))

# Alert System Settings
st.sidebar.subheader("⚙️ Alert Settings")
enable_alerts = st.sidebar.checkbox("Enable Alert System", value=True)
alert_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
student_id = st.sidebar.text_input("Student ID (Optional)", "")

# Initialize placeholders for displaying metrics
cheating_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()
alert_log_placeholder = st.sidebar.empty()


class YOLOVideoProcessor(VideoProcessorBase):
    """Processes webcam frames from the browser using YOLO detection."""

    def __init__(self):
        loaded = load_models()
        self.model = (
            loaded.get("YOLOv8 Full (New)") or
            loaded.get("YOLOv8 Standard") or
            list(loaded.values())[0]
        )
        self.model_names = self.model.names  # dict {id: class_name} from the actual model
        self.cheating_classes = get_cheating_classes(self.model_names)
        self.alert_threshold = 0.25
        self.enable_alerts = True
        self.student_id = ""
        self._lock = threading.Lock()
        self.cheating_count = 0
        self.last_alerts = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        results = self.model(img, conf=self.alert_threshold, verbose=False)
        cheat_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = self.model_names.get(class_id, f'class_{class_id}')
                conf = float(box.conf)

                is_cheating = class_name in self.cheating_classes
                color = (0, 0, 255) if is_cheating else (0, 255, 0)

                if is_cheating:
                    cheat_count += 1
                    if self.enable_alerts and alert_system.should_alert(class_name, conf):
                        record = alert_system.save_alert_frame(
                            img, class_name, conf, self.student_id or None
                        )
                        img = AlertNotifier.create_visual_alert(img, class_name, conf)
                        with self._lock:
                            self.last_alerts.append({
                                'time': time.time(),
                                'class': class_name,
                                'conf': conf,
                                'record': record
                            })
                            self.last_alerts = self.last_alerts[-10:]

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        with self._lock:
            self.cheating_count = cheat_count

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def update_class_counts(results, model):
    model_names = model.names
    counts = {name: 0 for name in model_names.values()}
    if results is None:
        return counts

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf)
                if confidence < alert_threshold:
                    continue
                class_id = int(box.cls)
                class_name = model_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
    return counts

def display_metrics(counts):
    cheat_count = sum(v for k, v in counts.items() if k in CHEATING_CLASSES_FULL)
    cheating_placeholder.metric(label='🚨 Suspicious Activities', value=cheat_count)
    if cheat_count > 0:
        status_placeholder.error("⚠️ ALERT: SUSPICIOUS ACTIVITY DETECTED")
    else:
        status_placeholder.success("✅ STATUS: NORMAL")

def process_video_capture(video_capture, stframe, model, model_selection):
    alert_log = st.empty()
    alerts = []
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=alert_threshold, verbose=False)
        if not results:
            continue

        counts = update_class_counts(results, model)
        display_metrics(counts)

        model_names = model.names
        cheating_cls = get_cheating_classes(model_names)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = model_names.get(class_id, f'class_{class_id}')
                conf = float(box.conf)

                is_cheating = class_name in cheating_cls
                color = (0, 0, 255) if is_cheating else (0, 255, 0)
                if is_cheating:
                    if enable_alerts and alert_system.should_alert(class_name, conf):
                        alert_record = alert_system.save_alert_frame(
                            frame, class_name, conf, student_id if student_id else None
                        )
                        alerts.append({
                            'time': time.time(),
                            'class': class_name,
                            'conf': conf,
                            'record': alert_record
                        })
                        frame = AlertNotifier.create_visual_alert(frame, class_name, conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f'{class_name} {conf:.2f}'
                cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if alerts:
            with alert_log.container():
                st.write("### 🚨 Recent Alerts")
                for a in alerts[-5:]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.warning(f"[{datetime.fromtimestamp(a['time']).strftime('%H:%M:%S')}] **{a['class']}** (Conf: {a['conf']:.2f})")
                    with col2:
                        if st.button("View", key=f"view_{a['record']['timestamp']}"):
                            st.image(a['record']['filepath'])

        annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels='RGB', use_column_width=True)

    return st.button("Replay Video")


def detection_page():
    st.title("📹 Real-time Detection")
    model_selection = st.sidebar.selectbox("Select YOLO model", list(model_paths.keys()))
    model = models[model_selection]

    # Wider video stream box
    st.markdown(
        """<style>
        div[data-testid="stVideo"] video,
        .stWebrtc video { width: 100% !important; max-width: 900px; }
        </style>""",
        unsafe_allow_html=True,
    )

    if source == 'Live Video':
        st.info("📷 Click **START** below to allow browser camera access and begin detection.")

        ctx = webrtc_streamer(
            key="exam-monitoring",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.model = model
            ctx.video_processor.model_names = model.names
            ctx.video_processor.cheating_classes = get_cheating_classes(model.names)
            ctx.video_processor.alert_threshold = alert_threshold
            ctx.video_processor.enable_alerts = enable_alerts
            ctx.video_processor.student_id = student_id

            with ctx.video_processor._lock:
                count = ctx.video_processor.cheating_count
                recent_alerts = ctx.video_processor.last_alerts.copy()

            cheating_placeholder.metric(label='🚨 Suspicious Activities', value=count)
            if count > 0:
                status_placeholder.error("⚠️ ALERT: SUSPICIOUS ACTIVITY DETECTED")
            else:
                status_placeholder.success("✅ STATUS: NORMAL")

            if recent_alerts:
                with alert_log_placeholder.container():
                    st.write("### 🚨 Recent Alerts")
                    for a in recent_alerts[-5:]:
                        st.warning(
                            f"[{datetime.fromtimestamp(a['time']).strftime('%H:%M:%S')}] "
                            f"**{a['class']}** (Conf: {a['conf']:.2f})"
                        )

    elif source == 'Upload MP4 File':
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
                tmpfile.write(uploaded_file.read())
                tmpfile_path = tmpfile.name
            stframe = st.empty()
            video_capture = cv2.VideoCapture(tmpfile_path)
            process_video_capture(video_capture, stframe, model, model_selection)
            video_capture.release()


def alert_dashboard_page():
    st.title("📊 Alert Dashboard")

    summary = alert_system.get_alert_summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Alerts", summary['total_alerts'])
    with col2:
        st.metric("Alert Types", len(summary['alerts_by_class']))
    with col3:
        st.metric("Recent Alerts", len(summary['recent_alerts']))

    if summary['alerts_by_class']:
        st.subheader("Alerts by Behavior")
        df = pd.DataFrame(list(summary['alerts_by_class'].items()), columns=['Behavior', 'Count'])

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df.set_index('Behavior'))

        with col2:
            st.write("### Behavior Details")
            for behavior, count in summary['alerts_by_class'].items():
                if behavior in CHEATING_BEHAVIORS:
                    info = CHEATING_BEHAVIORS[behavior]
                    severity = info['severity'].upper()
                    st.write(f"**{behavior}** ({severity}): {count} alerts")
                    st.caption(info['description'])

    if summary['recent_alerts']:
        st.subheader("Recent Alerts")
        for alert in summary['recent_alerts'][-10:]:
            with st.expander(f"{alert['class']} - {alert['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Confidence:** {alert['confidence']:.2f}")
                    st.write(f"**Time:** {alert['timestamp']}")
                with col2:
                    if os.path.exists(alert['filepath']):
                        st.image(alert['filepath'], caption=alert['class'])

    if st.button("📄 Generate Alert Report"):
        report = alert_system.generate_alert_report()
        st.download_button(
            label="Download Report",
            data=report,
            file_name="alert_report.txt",
            mime="text/plain"
        )


def comparison_page():
    st.title("📈 Model Comparison & Metrics")
    st.write("Detailed comparison and metrics will be available after full training is complete.")

    st.subheader("Model Information")
    st.write("""
    - **YOLOv8 Full (New)**: Trained on 2,700+ images with 50 epochs
    - **Classes**: 13 (12 cheating behaviors + Face detection)
    - **mAP50**: ~0.81 (81% accuracy)
    - **mAP50-95**: ~0.57
    """)


# Display appropriate page based on selection
if page == 'Detection':
    detection_page()
elif page == 'Alert Dashboard':
    alert_dashboard_page()
elif page == 'Comparison':
    comparison_page()
