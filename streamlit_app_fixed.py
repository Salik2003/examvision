# ExamVision - Real-Time Exam Monitoring System

import streamlit as st
import cv2
import os
import json
import time
import threading
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime
from ultralytics import YOLO
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
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #ffffff !important; }
    h1 { color: #1e293b !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    h2, h3 { color: #334155 !important; font-weight: 600 !important; }
    .metric-card {
        background: #ffffff; border-radius: 16px; padding: 24px; text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 12px 28px rgba(0,0,0,0.08);
        transform: translateY(-4px);
    }
    .metric-value { font-size: 38px; font-weight: 700; margin: 8px 0; letter-spacing: -1px; }
    .metric-label { font-size: 12px; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-sublabel { font-size: 12px; color: #64748b; margin-top: 4px; }
    .red-val { color: #ef4444; }
    .blue-val { color: #3b82f6; }
    .green-val { color: #10b981; }
    .orange-val { color: #f59e0b; }
    .breakdown-card {
        background: #ffffff; border-radius: 12px; padding: 16px; border-left: 5px solid;
        margin: 4px 0; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: all 0.3s ease;
    }
    .breakdown-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px); }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1e293b !important;
        margin: 24px 0 12px 0; padding-bottom: 10px;
        border-bottom: 3px solid #3b82f6; display: inline-block;
    }
    .evidence-caption { font-size: 11px; color: #64748b; text-align: center; margin-top: 4px; }
    .dashboard-title { font-size: 28px; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
    .dashboard-subtitle { font-size: 14px; color: #64748b; margin-bottom: 24px; }
    div[data-testid="stMetric"] {
        background: #ffffff; border-radius: 16px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 6px 16px rgba(0,0,0,0.04);
    }
    .stButton > button {
        border-radius: 10px; font-weight: 600; padding: 10px 28px;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important; border: none; font-size: 14px; transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 16px rgba(59,130,246,0.4); transform: translateY(-2px);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important; border: none !important; border-radius: 10px; font-weight: 600;
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

# Classes from the trained 50-epoch model
CLASS_NAMES = {
    0: 'look_down', 1: 'look_forward', 2: 'look_left', 3: 'look_right', 4: 'look_up',
    5: 'mouth_close', 6: 'mouth_open', 7: 'see_down', 8: 'see_forward', 9: 'see_left',
    10: 'see_right', 11: 'see_up', 12: 'Face'
}

CHEATING_CLASSES = {
    'look_left', 'look_right', 'look_up', 'look_down',
    'mouth_open', 'see_left', 'see_right', 'see_down'
}

COCO_PHONE_CLASS_ID = 67

# -----------------------------------------------
# Alert Helpers
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
    filename = f"{class_name}_{ts_str}_{confidence:.2f}.jpg"
    filepath = os.path.join("cheating_detections", filename)
    cv2.imwrite(filepath, frame)
    alerts = load_alerts()
    alerts.append({
        "timestamp": ts.isoformat(),
        "class": class_name,
        "confidence": float(confidence),
        "filepath": filepath,
        "student_id": student_id if student_id else None
    })
    save_alerts(alerts)


# -----------------------------------------------
# Model Loading
# -----------------------------------------------
@st.cache_resource
def load_behavior_model():
    path = 'yolov8/trained_model_50_epochs.pt'
    return YOLO(path) if os.path.exists(path) else YOLO('yolov8n.pt')


@st.cache_resource
def load_phone_model():
    return YOLO("yolov8n.pt")


# -----------------------------------------------
# WebRTC Processor
# -----------------------------------------------
class ExamVisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.behavior_model = load_behavior_model()
        self.phone_model = load_phone_model()
        self.conf_thresh = 0.35
        self.phone_conf_thresh = 0.15
        self.enable_phone = True
        self.student_id = ""
        self._lock = threading.Lock()
        self.suspicious_count = 0
        self._last_alert_time = {}
        self.cooldown_seconds = 3

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        suspicious = 0
        now = time.time()

        # Behavior detection (look_left, look_right, etc.)
        results = self.behavior_model(img, conf=self.conf_thresh, verbose=False, imgsz=320)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                class_name = self.behavior_model.names.get(class_id, CLASS_NAMES.get(class_id, f'class_{class_id}'))
                conf = float(box.conf)
                is_cheating = class_name in CHEATING_CLASSES
                color = (0, 0, 255) if is_cheating else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{class_name} {conf:.2f}', (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if is_cheating:
                    suspicious += 1
                    if (now - self._last_alert_time.get(class_name, 0)) >= self.cooldown_seconds:
                        append_alert(class_name, conf, img, self.student_id)
                        self._last_alert_time[class_name] = now

        # Phone detection
        if self.enable_phone:
            phone_results = self.phone_model(img, conf=self.phone_conf_thresh,
                                              classes=[COCO_PHONE_CLASS_ID], verbose=False)
            for result in phone_results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(img, f'PHONE {conf:.2f}', (x1, max(y1 - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    suspicious += 1
                    if (now - self._last_alert_time.get('phone', 0)) >= self.cooldown_seconds:
                        append_alert('phone', conf, img, self.student_id)
                        self._last_alert_time['phone'] = now

        with self._lock:
            self.suspicious_count = suspicious

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------------------------
# Sidebar Controls
# -----------------------------------------------
st.sidebar.header("Controls")
page = st.sidebar.radio("Select Page", ["Detection", "Alert Dashboard"])

st.sidebar.subheader("Phone Detection")
phone_conf_thresh = st.sidebar.slider("Phone Confidence Threshold", 0.0, 1.0, 0.15, 0.05)
enable_phone = st.sidebar.checkbox("Enable Phone Detection", value=True)

st.sidebar.subheader("Detection Settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
source = st.sidebar.radio("Source", ["Webcam", "Upload Video"])
student_id = st.sidebar.text_input("Student ID (Optional)")

if enable_phone:
    st.sidebar.success("Phone detection: ON")
else:
    st.sidebar.warning("Phone detection: OFF")
st.sidebar.success("Behavior detection: ON (YOLO)")

# -----------------------------------------------
# Detection Page
# -----------------------------------------------
if page == "Detection":
    st.subheader("Live Detection")

    if source == "Webcam":
        st.info("📷 Click **START** to allow browser camera access and begin detection.")

        ctx = webrtc_streamer(
            key="examvision",
            video_processor_factory=ExamVisionProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.conf_thresh = conf_thresh
            ctx.video_processor.phone_conf_thresh = phone_conf_thresh
            ctx.video_processor.enable_phone = enable_phone
            ctx.video_processor.student_id = student_id

            with ctx.video_processor._lock:
                count = ctx.video_processor.suspicious_count

            c1, c2 = st.columns(2)
            c1.metric("Suspicious (current frame)", count)
            c2.metric("Status", "🚨 ALERT" if count > 0 else "✅ Normal")

    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            ext = os.path.splitext(uploaded_file.name)[1]
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tfile.write(uploaded_file.read())
            tfile.close()

            if st.button("Start Video Detection"):
                behavior_model = load_behavior_model()
                phone_model = load_phone_model()
                cap = cv2.VideoCapture(tfile.name)
                frame_ph = st.empty()
                stop_btn = st.button("Stop")
                last_alert_time = {}

                while cap.isOpened() and not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    now = time.time()
                    results = behavior_model(frame, conf=conf_thresh, verbose=False, imgsz=320)
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            class_id = int(box.cls)
                            class_name = behavior_model.names.get(class_id, CLASS_NAMES.get(class_id, ''))
                            conf = float(box.conf)
                            is_cheating = class_name in CHEATING_CLASSES
                            color = (0, 0, 255) if is_cheating else (0, 255, 0)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, max(y1 - 8, 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            if is_cheating and (now - last_alert_time.get(class_name, 0)) >= 3:
                                append_alert(class_name, conf, frame, student_id)
                                last_alert_time[class_name] = now

                    if enable_phone:
                        phone_res = phone_model(frame, conf=phone_conf_thresh,
                                                classes=[COCO_PHONE_CLASS_ID], verbose=False)
                        for result in phone_res:
                            if result.boxes is None:
                                continue
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(frame, f'PHONE {conf:.2f}', (x1, max(y1 - 10, 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                if (now - last_alert_time.get('phone', 0)) >= 3:
                                    append_alert('phone', conf, frame, student_id)
                                    last_alert_time['phone'] = now

                    frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                   channels="RGB", use_container_width=True)

                cap.release()
                st.info("Detection stopped.")

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

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Alerts</div><div class="metric-value red-val">{total_alerts}</div><div class="metric-sublabel">All suspicious activities</div></div>', unsafe_allow_html=True)
        with m2:
            unique_types = df["class"].nunique() if "class" in df.columns else 0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Unique Cheating Types</div><div class="metric-value blue-val">{unique_types}</div><div class="metric-sublabel">Different behaviors detected</div></div>', unsafe_allow_html=True)
        with m3:
            if "class" in df.columns and len(df) > 0:
                most_common = df["class"].value_counts().index[0]
                most_count = df["class"].value_counts().values[0]
                pct = f"{(most_count / total_alerts) * 100:.1f}"
            else:
                most_common, most_count, pct = "N/A", 0, "0"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Most Common</div><div class="metric-value green-val">{most_common}</div><div class="metric-sublabel">{most_count} alerts ({pct}%)</div></div>', unsafe_allow_html=True)
        with m4:
            if "timestamp" in df.columns and len(df) > 0:
                first_t = df["timestamp"].min().strftime("%H:%M:%S")
                last_t = df["timestamp"].max().strftime("%H:%M:%S")
                duration = df["timestamp"].max() - df["timestamp"].min()
                mins = int(duration.total_seconds() // 60)
                secs = int(duration.total_seconds() % 60)
                dur_str = f"{mins}m {secs}s"
            else:
                first_t, last_t, dur_str = "N/A", "N/A", "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Time Range</div><div class="metric-value orange-val">{first_t} – {last_t}</div><div class="metric-sublabel">Duration: {dur_str}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown('<div class="section-header">Alerts by Type</div>', unsafe_allow_html=True)
            if "class" in df.columns:
                cc = df["class"].value_counts().reset_index()
                cc.columns = ["Class", "Count"]
                st.bar_chart(cc.set_index("Class"), color="#3b82f6")
        with chart2:
            st.markdown('<div class="section-header">Alerts Over Time</div>', unsafe_allow_html=True)
            if "timestamp" in df.columns:
                df["minute"] = df["timestamp"].dt.floor("min")
                tc = df.groupby("minute").size().reset_index()
                tc.columns = ["Time", "Count"]
                st.line_chart(tc.set_index("Time"), color="#ef4444")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Cheating Breakdown</div>', unsafe_allow_html=True)
        if "class" in df.columns:
            classes = df["class"].value_counts()
            colors = ["#ef4444", "#f97316", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
            bcols = st.columns(len(classes))
            for i, (cls_name, count) in enumerate(classes.items()):
                pct = f"{(count / total_alerts) * 100:.1f}"
                c = colors[i % len(colors)]
                with bcols[i]:
                    st.markdown(f'<div class="breakdown-card" style="border-left-color:{c};"><div style="color:{c};font-size:11px;font-weight:700;text-transform:uppercase;">{cls_name}</div><div style="font-size:32px;font-weight:700;color:#1e293b;margin:4px 0;">{count}</div><div style="font-size:12px;color:#64748b;">{pct}%</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
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
            else:
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
                for _, row in filtered_df.iterrows():
                    fp = str(row.get("filepath", ""))
                    if os.path.exists(fp):
                        img = cv2.imread(fp)
                        if img is not None:
                            with img_cols[shown % 4]:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                         caption=f"{row.get('class','')} {float(row.get('confidence',0)):.2f}",
                                         use_container_width=True)
                            shown += 1
                    if shown >= 12:
                        break
                if shown == 0:
                    st.info("No evidence images found.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Actions</div>', unsafe_allow_html=True)
        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            st.download_button("Download CSV Report", df.to_csv(index=False),
                               "exam_alerts.csv", "text/csv", use_container_width=True)
        with btn2:
            st.download_button("Download JSON Report", json.dumps(alerts, indent=2),
                               "exam_alerts.json", "application/json", use_container_width=True)
        with btn3:
            if st.button("Clear All Alerts", use_container_width=True):
                save_alerts([])
                for f in os.listdir("cheating_detections"):
                    try:
                        os.remove(os.path.join("cheating_detections", f))
                    except Exception:
                        pass
                st.success("Cleared! Refresh the page.")
    else:
        st.markdown('<div style="text-align:center;padding:100px 20px;"><div style="font-size:64px;margin-bottom:16px;">📊</div><div style="font-size:24px;font-weight:600;color:#334155;margin-bottom:8px;">No alerts recorded yet</div><div style="font-size:14px;color:#94a3b8;">Run detection first to generate alerts</div></div>', unsafe_allow_html=True)
