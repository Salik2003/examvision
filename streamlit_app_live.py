import streamlit as st
import cv2
from ultralytics import YOLO
import utils.detection as detection
from utils.alert_system import AlertSystem, AlertNotifier, CHEATING_BEHAVIORS
import tempfile
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Create detections folder if it doesn't exist
os.makedirs("cheating_detections", exist_ok=True)

# Initialize Alert System
alert_system = AlertSystem()

# Define class names mapping for the new model (13 classes)
class_names = {
    0: 'look_down', 1: 'look_forward', 2: 'look_left', 3: 'look_right', 4: 'look_up',
    5: 'mouth_close', 6: 'mouth_open', 7: 'see_down', 8: 'see_forward', 9: 'see_left',
    10: 'see_right', 11: 'see_up', 12: 'Face'
}

# Cheating classes that trigger alerts
CHEATING_CLASSES = ['look_left', 'look_right', 'look_up', 'look_down', 'mouth_open', 'see_left', 'see_right', 'see_down']

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
st.set_page_config(page_title="Exam Monitoring System - Live", layout="wide")
st.sidebar.title('🎓 Exam Monitoring System')

# Page selection
page = st.sidebar.radio("Select Page", ('Detection', 'Alert Dashboard'))

# Alert System Settings
st.sidebar.subheader("⚙️ Alert Settings")
enable_alerts = st.sidebar.checkbox("Enable Alert System", value=True)
alert_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
student_id = st.sidebar.text_input("Student ID (Optional)", "")

# Initialize placeholders for displaying metrics
cheating_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()

def update_class_counts(results):
    counts = {name: 0 for name in class_names.values()}
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
                class_name = class_names.get(class_id, 'Unknown')
                if class_name in counts:
                    counts[class_name] += 1
    return counts

def display_metrics(counts):
    cheat_count = sum(counts[cls] for cls in CHEATING_CLASSES if cls in counts)
    cheating_placeholder.metric(label='🚨 Suspicious Activities', value=cheat_count)
    if cheat_count > 0:
        status_placeholder.error("⚠️ ALERT: SUSPICIOUS ACTIVITY")
    else:
        status_placeholder.success("✅ STATUS: NORMAL")

def detection_page():
    st.title("📹 Real-time Live Detection")
    
    model_selection = st.sidebar.selectbox("Select YOLO model", list(model_paths.keys()))
    model = models[model_selection]
    
    source = st.radio("Select Source", ("Webcam (Local)", "Upload Video"))
    
    if source == "Webcam (Local)":
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        
        alerts = []
        alert_log_container = st.container()

        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access webcam.")
                break
            
            # Run Detection
            results = model(frame, conf=alert_threshold, verbose=False)
            counts = update_class_counts(results)
            display_metrics(counts)
            
            # Annotate frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    class_name = class_names.get(class_id, f'Class {class_id}')
                    conf = float(box.conf)
                    
                    color = (0, 255, 0)
                    if class_name in CHEATING_CLASSES:
                        color = (0, 0, 255)
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
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display Alert Log
            if alerts:
                with alert_log_container:
                    st.write("### 🚨 Recent Alerts")
                    for a in alerts[-3:]:
                        st.warning(f"[{datetime.fromtimestamp(a['time']).strftime('%H:%M:%S')}] **{a['class']}**")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        else:
            st.write('Stopped')
            camera.release()

    elif source == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                results = model(frame, conf=alert_threshold, verbose=False)
                counts = update_class_counts(results)
                display_metrics(counts)
                
                # Annotate (Simplified for brevity)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = class_names.get(int(box.cls), 'Object')
                        color = (0, 0, 255) if class_name in CHEATING_CLASSES else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            vf.release()

def alert_dashboard_page():
    st.title("📊 Alert Dashboard")
    summary = alert_system.get_alert_summary()
    st.write(f"Total Alerts: {summary['total_alerts']}")
    # ... (Rest of the dashboard code from original app)

if page == 'Detection':
    detection_page()
elif page == 'Alert Dashboard':
    alert_dashboard_page()
