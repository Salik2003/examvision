# ExamVision - Fixed Real-Time Exam Monitoring System

import streamlit as st
import cv2
import os
import json
import time
import math
import threading
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# -----------------------------------------------
# App Configuration
# -----------------------------------------------
st.set_page_config(page_title="ExamVision - Exam Monitoring System", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f0 100%);
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label {
        color: #94a3b8 !important;
        font-size: 13px;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    h1 {
        color: #1e293b !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: none;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 12px 28px rgba(0,0,0,0.08);
        transform: translateY(-4px);
    }
    .metric-value {
        font-size: 38px;
        font-weight: 700;
        margin: 8px 0;
        letter-spacing: -1px;
    }
    .metric-label {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-sublabel {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    .red-val { color: #ef4444; }
    .blue-val { color: #3b82f6; }
    .green-val { color: #10b981; }
    .orange-val { color: #f59e0b; }
    .breakdown-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid;
        margin: 4px 0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .breakdown-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b !important;
        margin: 24px 0 12px 0;
        padding-bottom: 10px;
        border-bottom: 3px solid #3b82f6;
        display: inline-block;
    }
    .evidence-caption {
        font-size: 11px;
        color: #64748b;
        text-align: center;
        margin-top: 4px;
    }
    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .dashboard-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 24px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: none;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 12px 28px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 13px;
        font-weight: 500;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #1e293b !important;
        font-weight: 700;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 28px;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        border: none;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 16px rgba(59,130,246,0.4);
        transform: translateY(-2px);
    }
    .stDataFrame {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stImage"] {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        overflow: hidden;
    }
    div[data-testid="stImage"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59,130,246,0.2);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 4px 16px rgba(16,185,129,0.4);
    }
    div.stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Powered Online Invigilation Assistant")

os.makedirs("cheating_detections", exist_ok=True)
ALERT_LOG_PATH = "alert_log.json"

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["turn:13.212.250.73:3478"], "username": "exam", "credential": "vision123"},
        {"urls": ["turn:13.212.250.73:3478?transport=tcp"], "username": "exam", "credential": "vision123"},
    ]
})
COCO_PHONE_CLASS_ID = 67

CHEATING_CLASSES = {
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "mouth_open",
    "phone",
    "eye_left",
    "eye_right",
    "eye_down"
}

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def load_alerts():
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_alerts(alerts):
    with open(ALERT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)


def append_alert(class_name, confidence, frame, student_id=""):
    ts = datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")
    conf_str = "{:.2f}".format(confidence)
    filename = class_name + "_" + ts_str + "_" + conf_str + ".jpg"
    filepath = os.path.join("cheating_detections", filename)
    cv2.imwrite(filepath, frame)

    alerts = load_alerts()
    alert_entry = {
        "timestamp": ts.isoformat(),
        "class": class_name,
        "confidence": float(confidence),
        "filepath": filepath,
        "student_id": student_id if student_id else None
    }
    alerts.append(alert_entry)
    save_alerts(alerts)


# -----------------------------------------------
# Model Loading
# -----------------------------------------------
@st.cache_resource
def load_coco_model():
    return YOLO("yolov8n.pt")


@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


# -----------------------------------------------
# Head Pose Estimation (using MediaPipe + solvePnP)
# -----------------------------------------------
def estimate_head_pose(landmarks, img_w, img_h):
    nose = landmarks[1]
    left_face = landmarks[234]
    right_face = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]

    # --- YAW (Left/Right) ---
    face_width = abs(right_face.x - left_face.x)
    if face_width < 0.001:
        yaw = 0.0
    else:
        face_center_x = (left_face.x + right_face.x) / 2.0
        yaw = ((nose.x - face_center_x) / face_width) * 180.0

    # --- PITCH (Up/Down) ---
    face_height = abs(chin.y - forehead.y)
    if face_height < 0.001:
        pitch = 0.0
    else:
        nose_ratio = (nose.y - forehead.y) / face_height
        pitch = (nose_ratio - 0.63) * -180.0

    return yaw, pitch
# -----------------------------------------------
# Mouth Open Detection
# -----------------------------------------------
def check_mouth_open(landmarks, img_w, img_h, threshold):
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_corner = landmarks[78]
    right_corner = landmarks[308]

    vertical = math.sqrt(
        ((upper_lip.x - lower_lip.x) * img_w) ** 2 +
        ((upper_lip.y - lower_lip.y) * img_h) ** 2
    )

    horizontal = math.sqrt(
        ((left_corner.x - right_corner.x) * img_w) ** 2 +
        ((left_corner.y - right_corner.y) * img_h) ** 2
    )

    if horizontal < 1.0:
        return False, 0.0

    ratio = vertical / horizontal
    return (ratio > threshold), ratio

    # -----------------------------------------------
# Eye Gaze Detection (using iris landmarks)
# -----------------------------------------------
def check_eye_gaze(landmarks, h_thresh, v_thresh):
    # Left eye: outer corner=33, inner corner=133, iris center=468
    # Right eye: inner corner=362, outer corner=263, iris center=473
    left_outer = landmarks[33]
    left_inner = landmarks[133]
    left_iris = landmarks[468]

    right_inner = landmarks[362]
    right_outer = landmarks[263]
    right_iris = landmarks[473]

    # Left eye horizontal ratio
    left_width = left_inner.x - left_outer.x
    if abs(left_width) < 0.001:
        left_h = 0.5
    else:
        left_h = (left_iris.x - left_outer.x) / left_width

    # Right eye horizontal ratio
    right_width = right_outer.x - right_inner.x
    if abs(right_width) < 0.001:
        right_h = 0.5
    else:
        right_h = (right_iris.x - right_inner.x) / right_width

    h_ratio = (left_h + right_h) / 2.0

    # Left eye vertical ratio
    left_top = landmarks[159]
    left_bottom = landmarks[145]
    left_height = left_bottom.y - left_top.y
    if abs(left_height) < 0.001:
        left_v = 0.5
    else:
        left_v = (left_iris.y - left_top.y) / left_height

    # Right eye vertical ratio
    right_top = landmarks[386]
    right_bottom = landmarks[374]
    right_height = right_bottom.y - right_top.y
    if abs(right_height) < 0.001:
        right_v = 0.5
    else:
        right_v = (right_iris.y - right_top.y) / right_height

    v_ratio = (left_v + right_v) / 2.0

    gaze_behaviors = []

    if h_ratio < (0.5 - h_thresh):
        gaze_behaviors.append("eye_left")
    elif h_ratio > (0.5 + h_thresh):
        gaze_behaviors.append("eye_right")

    if v_ratio > (0.5 + v_thresh):
        gaze_behaviors.append("eye_down")

    # Also check if upper eyelid covers iris (eyes drooping down)
    left_upper_dist = abs(left_iris.y - landmarks[159].y)
    right_upper_dist = abs(right_iris.y - landmarks[386].y)
    left_lower_dist = abs(landmarks[145].y - left_iris.y)
    right_lower_dist = abs(landmarks[374].y - right_iris.y)

    if left_lower_dist > 0.001 and right_lower_dist > 0.001:
        left_ratio = left_upper_dist / left_lower_dist
        right_ratio = right_upper_dist / right_lower_dist
        avg_ratio = (left_ratio + right_ratio) / 2.0
        if avg_ratio > 2.0 and "eye_down" not in gaze_behaviors:
            gaze_behaviors.append("eye_down")

    return gaze_behaviors, h_ratio, v_ratio


# -----------------------------------------------
# MediaPipe Frame Processing
# -----------------------------------------------
def process_frame_mediapipe(frame, face_mesh, yaw_thresh, pitch_thresh, mouth_thresh, eye_h_thresh=0.15, eye_v_thresh=0.15):
    img_h, img_w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True

    cheating_detections = []
    suspicious_count = 0

    if results.multi_face_landmarks is None:
        return frame, cheating_detections, suspicious_count

    for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
        lm = face_landmarks.landmark
        student_label = "Student " + str(face_idx + 1)

        try:
            yaw_deg, pitch_deg = estimate_head_pose(lm, img_w, img_h)
        except Exception:
            yaw_deg = 0.0
            pitch_deg = 0.0

        try:
            mouth_flag, mouth_ratio = check_mouth_open(lm, img_w, img_h, mouth_thresh)
        except Exception:
            mouth_flag = False
            mouth_ratio = 0.0

        behaviors = []

        if yaw_deg < -yaw_thresh:
            behaviors.append("look_left")
        elif yaw_deg > yaw_thresh:
            behaviors.append("look_right")

        if pitch_deg < -pitch_thresh:
            behaviors.append("look_down")
        elif pitch_deg > pitch_thresh:
            behaviors.append("look_up")

        if mouth_flag:
            behaviors.append("mouth_open")

        # Eye gaze detection
        try:
            gaze_behaviors, h_ratio, v_ratio = check_eye_gaze(lm, eye_h_thresh, eye_v_thresh)
            behaviors.extend(gaze_behaviors)
        except Exception:
            pass

        is_cheating = len(behaviors) > 0

        x_coords = [lm[i].x for i in range(468)]
        y_coords = [lm[i].y for i in range(468)]
        x_min = int(min(x_coords) * img_w) - 10
        y_min = int(min(y_coords) * img_h) - 10
        x_max = int(max(x_coords) * img_w) + 10
        y_max = int(max(y_coords) * img_h) + 10
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        if is_cheating:
            box_color = (0, 0, 255)
        else:
            box_color = (0, 255, 0)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)

        yaw_text = "Yaw: " + "{:.1f}".format(yaw_deg)
        pitch_text = "Pitch: " + "{:.1f}".format(pitch_deg)
        cv2.putText(frame, yaw_text, (x_min, max(20, y_min - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        cv2.putText(frame, pitch_text, (x_min, max(20, y_min - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        if is_cheating:
            cv2.putText(frame, "Cheating", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            suspicious_count += 1
            for b in behaviors:
                cheating_detections.append({
                    "class": b,
                    "confidence": 1.0,
                    "box": (x_min, y_min, x_max, y_max)
                })
        else:
            cv2.putText(frame, "Normal", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, cheating_detections, suspicious_count


# -----------------------------------------------
# Phone Detection (COCO YOLOv8n)
# -----------------------------------------------
def run_phone_inference(coco_model, frame, conf_thresh):
    results = coco_model(frame, verbose=False, conf=conf_thresh, classes=[COCO_PHONE_CLASS_ID])
    detected = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()
            x1 = int(coords[0])
            y1 = int(coords[1])
            x2 = int(coords[2])
            y2 = int(coords[3])
            detected.append({
                "class": "phone",
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })
    return detected


def draw_phone_boxes(frame, phone_detections):
    for d in phone_detections:
        x1, y1, x2, y2 = d["box"]
        conf = d["confidence"]
        label = "PHONE " + "{:.2f}".format(conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame


# -----------------------------------------------
# Main Video Processing Loop
# -----------------------------------------------
def process_video_capture(video_capture, face_mesh, coco_model, phone_conf_thresh,
                          yaw_thresh, pitch_thresh, mouth_thresh,
                          display_fps, student_id="", phone_detection=True,
                          eye_h_thresh=0.15, eye_v_thresh=0.15):
    frame_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)
    live_alerts_box = st.empty()

    total_suspicious = 0
    frame_counter = 0
    start_time = time.time()

    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = {}
    cooldown_seconds = 3

    stop_btn = st.button("Stop Detection")

    while video_capture.isOpened():
        if stop_btn:
            break

        ret, frame = video_capture.read()
        if not ret:
            break

        frame_counter += 1

        annotated_frame, head_alerts, head_suspicious = process_frame_mediapipe(
            frame, face_mesh, yaw_thresh, pitch_thresh, mouth_thresh, eye_h_thresh, eye_v_thresh
        )

        phone_alerts = []
        if phone_detection and coco_model is not None:
            phone_alerts = run_phone_inference(coco_model, annotated_frame, phone_conf_thresh)
            annotated_frame = draw_phone_boxes(annotated_frame, phone_alerts)

        all_alerts = head_alerts + phone_alerts
        suspicious_count = head_suspicious + len(phone_alerts)
        total_suspicious += suspicious_count

        now = time.time()
        for a in all_alerts:
            cls_name = a["class"]
            last_time = st.session_state.last_alert_time.get(cls_name, 0)
            if (now - last_time) >= cooldown_seconds:
                append_alert(cls_name, a["confidence"], annotated_frame, student_id)
                st.session_state.last_alert_time[cls_name] = now

        elapsed = time.time() - start_time
        if elapsed < 0.001:
            elapsed = 0.001
        fps = frame_counter / elapsed

        rgb_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_display, channels="RGB", use_container_width=True)

        col1.metric("Current Suspicious", suspicious_count)
        col2.metric("Total Suspicious", total_suspicious)
        if display_fps:
            col3.metric("FPS", "{:.1f}".format(fps))
        else:
            col3.metric("FPS", "Hidden")

        if all_alerts:
            alert_lines = []
            for a in all_alerts:
                line = "ALERT: " + a["class"] + " (conf: " + "{:.2f}".format(a["confidence"]) + ")"
                alert_lines.append(line)
            alert_text = " | ".join(alert_lines)
            live_alerts_box.error(alert_text)
        else:
            live_alerts_box.success("No suspicious activity in current frame")

    video_capture.release()
    st.info("Detection stopped.")


# -----------------------------------------------
# WebRTC Video Processor (MediaPipe + Phone)
# -----------------------------------------------
class MediaPipeVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=5, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.coco_model = None
        self.phone_detection = True
        self.phone_conf_thresh = 0.15
        self.yaw_thresh = 15
        self.pitch_thresh = 10
        self.mouth_thresh = 0.04
        self.eye_h_thresh = 0.15
        self.eye_v_thresh = 0.06
        self.student_id = ""
        self._lock = threading.Lock()
        self.suspicious_count = 0
        self._last_alert_time = {}
        self.cooldown_seconds = 3

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        annotated, cheating_detections, head_suspicious = process_frame_mediapipe(
            img, self.face_mesh, self.yaw_thresh, self.pitch_thresh,
            self.mouth_thresh, self.eye_h_thresh, self.eye_v_thresh
        )

        phone_detections = []
        if self.phone_detection and self.coco_model is not None:
            phone_detections = run_phone_inference(self.coco_model, annotated, self.phone_conf_thresh)
            annotated = draw_phone_boxes(annotated, phone_detections)

        all_detections = cheating_detections + phone_detections
        now = time.time()
        for a in all_detections:
            cls_name = a["class"]
            if (now - self._last_alert_time.get(cls_name, 0)) >= self.cooldown_seconds:
                append_alert(cls_name, a["confidence"], annotated, self.student_id)
                self._last_alert_time[cls_name] = now

        with self._lock:
            self.suspicious_count = head_suspicious + len(phone_detections)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# -----------------------------------------------
# Sidebar Controls
# -----------------------------------------------
st.sidebar.header("Controls")

page = st.sidebar.radio("Select Page", ["Detection", "Alert Dashboard"])

st.sidebar.subheader("Phone Detection")
phone_conf_thresh = st.sidebar.slider("Phone Confidence Threshold", 0.0, 1.0, 0.15, 0.05)
enable_phone = st.sidebar.checkbox("Enable Phone Detection", value=True)

st.sidebar.subheader("Head Pose Thresholds")
yaw_thresh = st.sidebar.slider("Left/Right Sensitivity (Yaw degrees)", 5, 45, 15, 5)
pitch_thresh = st.sidebar.slider("Up/Down Sensitivity (Pitch degrees)", 5, 40, 10, 5)

st.sidebar.subheader("Mouth Detection")
mouth_thresh = st.sidebar.slider("Mouth Open Threshold", 0.01, 0.15, 0.04, 0.005)

st.sidebar.subheader("Eye Gaze Detection")
eye_h_thresh = st.sidebar.slider("Eye Left/Right Sensitivity", 0.05, 0.30, 0.15, 0.05)
eye_v_thresh = st.sidebar.slider("Eye Down Sensitivity", 0.02, 0.30, 0.06, 0.02)

st.sidebar.subheader("General Settings")
source = st.sidebar.radio("Source", ["Webcam", "Upload Video"])
student_id = st.sidebar.text_input("Student ID (Optional)")
display_fps = st.sidebar.checkbox("Show FPS", value=True)

coco_model = None
if enable_phone:
    coco_model = load_coco_model()
    st.sidebar.success("Phone detection: ON")
else:
    st.sidebar.warning("Phone detection: OFF")

face_mesh = get_face_mesh()
st.sidebar.success("Head Pose detection: ON (MediaPipe)")


# -----------------------------------------------
# Detection Page
# -----------------------------------------------
if page == "Detection":
    st.subheader("Live Detection")

    if source == "Webcam":
        st.info("📷 Click **START** below to allow browser camera access and begin detection.")

        ctx = webrtc_streamer(
            key="examvision-mediapipe",
            video_processor_factory=MediaPipeVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.yaw_thresh = yaw_thresh
            ctx.video_processor.pitch_thresh = pitch_thresh
            ctx.video_processor.mouth_thresh = mouth_thresh
            ctx.video_processor.eye_h_thresh = eye_h_thresh
            ctx.video_processor.eye_v_thresh = eye_v_thresh
            ctx.video_processor.phone_detection = enable_phone
            ctx.video_processor.phone_conf_thresh = phone_conf_thresh
            ctx.video_processor.student_id = student_id
            if enable_phone and ctx.video_processor.coco_model is None:
                ctx.video_processor.coco_model = load_coco_model()

            with ctx.video_processor._lock:
                count = ctx.video_processor.suspicious_count

            c1, c2 = st.columns(2)
            c1.metric("Suspicious Activities (current frame)", count)
            c2.metric("Status", "🚨 ALERT" if count > 0 else "✅ Normal")
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            file_ext = os.path.splitext(uploaded_file.name)[1]
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            tfile.write(uploaded_file.read())
            tfile.close()

            start_btn = st.button("Start Video Detection")
            if start_btn:
                cap = cv2.VideoCapture(tfile.name)
                if not cap.isOpened():
                    st.error("Could not open the uploaded video file.")
                else:
                    process_video_capture(
                        cap, face_mesh, coco_model, phone_conf_thresh,
                        yaw_thresh, pitch_thresh, mouth_thresh,
                        display_fps, student_id=student_id,
                        phone_detection=enable_phone
                    )


# -----------------------------------------------
# Alert Dashboard Page
# -----------------------------------------------
elif page == "Alert Dashboard":
    st.markdown('<div class="dashboard-title">Alert Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Real-time overview of all suspicious activities detected during the exam</div>', unsafe_allow_html=True)
    alerts = load_alerts()

    if alerts:
        df = pd.DataFrame(alerts)
        total_alerts = len(df)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # --- Row 1: Metric Cards ---
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown('<div class="metric-card"><div class="metric-label">Total Alerts</div><div class="metric-value red-val">' + str(total_alerts) + '</div><div class="metric-sublabel">All suspicious activities</div></div>', unsafe_allow_html=True)

        with m2:
            unique_types = df["class"].nunique() if "class" in df.columns else 0
            st.markdown('<div class="metric-card"><div class="metric-label">Unique Cheating Types</div><div class="metric-value blue-val">' + str(unique_types) + '</div><div class="metric-sublabel">Different behaviors detected</div></div>', unsafe_allow_html=True)
        with m3:
            if "class" in df.columns and len(df) > 0:
                most_common = df["class"].value_counts().index[0]
                most_count = str(df["class"].value_counts().values[0])
                pct = "{:.1f}".format((df["class"].value_counts().values[0] / total_alerts) * 100)
            else:
                most_common = "N/A"
                most_count = "0"
                pct = "0"
            st.markdown('<div class="metric-card"><div class="metric-label">Most Common</div><div class="metric-value green-val">' + most_common + '</div><div class="metric-sublabel">' + most_count + ' alerts (' + pct + '%)</div></div>', unsafe_allow_html=True)

        with m4:
            if "timestamp" in df.columns and len(df) > 0:
                first_t = df["timestamp"].min().strftime("%H:%M:%S")
                last_t = df["timestamp"].max().strftime("%H:%M:%S")
                duration = df["timestamp"].max() - df["timestamp"].min()
                mins = int(duration.total_seconds() // 60)
                secs = int(duration.total_seconds() % 60)
                dur_str = str(mins) + "m " + str(secs) + "s"
            else:
                first_t = "N/A"
                last_t = "N/A"
                dur_str = "N/A"
            st.markdown('<div class="metric-card"><div class="metric-label">Time Range</div><div class="metric-value orange-val">' + first_t + ' - ' + last_t + '</div><div class="metric-sublabel">Duration: ' + dur_str + '</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 2: Charts ---
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown('<div class="section-header">Alerts by Type</div>', unsafe_allow_html=True)
            if "class" in df.columns:
                class_counts = df["class"].value_counts().reset_index()
                class_counts.columns = ["Class", "Count"]
                st.bar_chart(class_counts.set_index("Class"), color="#3b82f6")

        with chart2:
            st.markdown('<div class="section-header">Alerts Over Time</div>', unsafe_allow_html=True)
            if "timestamp" in df.columns:
                df["minute"] = df["timestamp"].dt.floor("min")
                time_counts = df.groupby("minute").size().reset_index()
                time_counts.columns = ["Time", "Count"]
                time_counts = time_counts.set_index("Time")
                st.line_chart(time_counts, color="#ef4444")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 3: Breakdown Cards ---
        st.markdown('<div class="section-header">Cheating Breakdown</div>', unsafe_allow_html=True)
        if "class" in df.columns:
            classes = df["class"].value_counts()
            colors = ["#ef4444", "#f97316", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
            breakdown_cols = st.columns(len(classes))
            for i, (cls_name, count) in enumerate(classes.items()):
                pct = "{:.1f}".format((count / total_alerts) * 100)
                col_color = colors[i % len(colors)]
                with breakdown_cols[i]:
                    st.markdown('<div class="breakdown-card" style="border-left-color: ' + col_color + ';"><div style="color: ' + col_color + '; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">' + cls_name + '</div><div style="font-size: 32px; font-weight: 700; color: #1e293b; margin: 4px 0;">' + str(count) + '</div><div style="font-size: 12px; color: #64748b; font-weight: 500;">' + pct + '%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 4: Table + Evidence ---
        left_panel, right_panel = st.columns([1, 1])

        with left_panel:
            st.markdown('<div class="section-header">Filter Alerts</div>', unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                all_classes = ["All"] + list(df["class"].unique()) if "class" in df.columns else ["All"]
                selected_class = st.selectbox("Filter by Type", all_classes)
            with f2:
                sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First", "Highest Confidence"])

            filtered_df = df.copy()
            if selected_class != "All":
                filtered_df = filtered_df[filtered_df["class"] == selected_class]

            if sort_order == "Newest First":
                filtered_df = filtered_df.sort_values("timestamp", ascending=False)
            elif sort_order == "Oldest First":
                filtered_df = filtered_df.sort_values("timestamp", ascending=True)
            elif sort_order == "Highest Confidence":
                filtered_df = filtered_df.sort_values("confidence", ascending=False)

            display_df = filtered_df[["timestamp", "class", "confidence", "student_id"]].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df.columns = ["Timestamp", "Class", "Confidence", "Student ID"]
            st.dataframe(display_df, use_container_width=True, height=400)

        with right_panel:
            st.markdown('<div class="section-header">Evidence Screenshots</div>', unsafe_allow_html=True)
            if "filepath" in filtered_df.columns:
                img_cols = st.columns(4)
                shown = 0
                for idx, row in filtered_df.iterrows():
                    fp = row.get("filepath", "")
                    if os.path.exists(str(fp)):
                        col_idx = shown % 4
                        with img_cols[col_idx]:
                            img = cv2.imread(str(fp))
                            if img is not None:
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                cls_str = str(row.get("class", ""))
                                conf_val = row.get("confidence", 0)
                                ts_short = row.get("timestamp")
                                if hasattr(ts_short, "strftime"):
                                    ts_short = ts_short.strftime("%H:%M:%S")
                                else:
                                    ts_short = str(ts_short)[:8]
                                cap = cls_str + " " + "{:.2f}".format(float(conf_val))
                                st.image(img_rgb, caption=cap, use_container_width=True)
                                st.markdown('<div class="evidence-caption">' + ts_short + '</div>', unsafe_allow_html=True)
                                shown += 1
                        if shown >= 12:
                            break
                if shown == 0:
                    st.info("No evidence images found.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Row 5: Actions ---
        st.markdown('<div class="section-header">Actions</div>', unsafe_allow_html=True)
        btn1, btn2, btn3 = st.columns(3)

        with btn1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv_data,
                file_name="exam_alerts.csv",
                mime="text/csv",
                use_container_width=True
            )

        with btn2:
            json_data = json.dumps(alerts, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name="exam_alerts.json",
                mime="application/json",
                use_container_width=True
            )

        with btn3:
            if st.button("Clear All Alerts", use_container_width=True):
                save_alerts([])
                if os.path.exists("cheating_detections"):
                    for f in os.listdir("cheating_detections"):
                        try:
                            os.remove(os.path.join("cheating_detections", f))
                        except Exception:
                            pass
                st.success("All alerts and evidence cleared! Refresh the page.")

    else:
        st.markdown('<div style="text-align: center; padding: 100px 20px;"><div style="font-size: 64px; margin-bottom: 16px;">📊</div><div style="font-size: 24px; font-weight: 600; color: #334155; margin-bottom: 8px;">No alerts recorded yet</div><div style="font-size: 14px; color: #94a3b8;">Run detection first to generate alerts and view analytics</div></div>', unsafe_allow_html=True)