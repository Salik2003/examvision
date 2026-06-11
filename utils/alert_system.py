"""
Alert Generation System for Exam Monitoring
Detects suspicious behavior and generates alerts with logging
"""

import cv2
import os
from datetime import datetime
from collections import defaultdict
import json


class AlertSystem:
    """
    Manages alert generation and logging for detected cheating behaviors
    """
    
    def __init__(self, alert_dir="cheating_detections", log_file="alert_log.json"):
        self.alert_dir = alert_dir
        self.log_file = log_file
        self.alerts = []
        self.alert_counts = defaultdict(int)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 2.0  # seconds between alerts of same type
        
        # Create directories
        os.makedirs(alert_dir, exist_ok=True)
        
        # Load existing alerts
        self._load_alert_log()
    
    def _load_alert_log(self):
        """Load existing alert log from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.alerts = json.load(f)
            except:
                self.alerts = []
    
    def _save_alert_log(self):
        """Save alert log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def should_alert(self, class_name, confidence):
        """
        Determine if an alert should be triggered based on cooldown
        """
        current_time = datetime.now().timestamp()
        last_time = self.last_alert_time.get(class_name, 0)
        
        if current_time - last_time > self.alert_cooldown:
            self.last_alert_time[class_name] = current_time
            return True
        return False
    
    def save_alert_frame(self, frame, class_name, confidence, student_id=None):
        """
        Save a frame with detected cheating behavior
        
        Args:
            frame: OpenCV image frame
            class_name: Name of detected class
            confidence: Detection confidence score
            student_id: Optional student identifier
        
        Returns:
            Alert record dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}_{confidence:.2f}.jpg"
        filepath = os.path.join(self.alert_dir, filename)
        
        # Save the frame
        cv2.imwrite(filepath, frame)
        
        # Create alert record
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'class': class_name,
            'confidence': float(confidence),
            'filepath': filepath,
            'student_id': student_id
        }
        
        self.alerts.append(alert_record)
        self.alert_counts[class_name] += 1
        self._save_alert_log()
        
        return alert_record
    
    def get_alert_summary(self):
        """
        Get summary of all alerts
        
        Returns:
            Dictionary with alert statistics
        """
        return {
            'total_alerts': len(self.alerts),
            'alerts_by_class': dict(self.alert_counts),
            'recent_alerts': self.alerts[-10:] if self.alerts else []
        }
    
    def get_alerts_for_student(self, student_id):
        """
        Get all alerts for a specific student
        """
        return [a for a in self.alerts if a.get('student_id') == student_id]
    
    def generate_alert_report(self, output_file="alert_report.txt"):
        """
        Generate a text report of all alerts
        """
        summary = self.get_alert_summary()
        
        report = f"""
EXAM MONITORING ALERT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
--------
Total Alerts: {summary['total_alerts']}

Alerts by Behavior:
"""
        for class_name, count in summary['alerts_by_class'].items():
            report += f"  - {class_name}: {count}\n"
        
        report += "\n\nRECENT ALERTS:\n"
        report += "-" * 50 + "\n"
        
        for alert in summary['recent_alerts'][-20:]:
            report += f"Time: {alert['timestamp']}\n"
            report += f"Behavior: {alert['class']}\n"
            report += f"Confidence: {alert['confidence']:.2f}\n"
            report += f"Evidence: {alert['filepath']}\n"
            report += "-" * 50 + "\n"
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        return report


class AlertNotifier:
    """
    Handles different alert notification methods
    """
    
    @staticmethod
    def create_visual_alert(frame, class_name, confidence):
        """
        Add visual alert overlay to frame
        """
        h, w = frame.shape[:2]
        
        # Red alert banner at top
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 255), -1)
        cv2.putText(frame, f"ALERT: {class_name.upper()}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Confidence score
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (w - 300, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def create_audio_alert():
        """
        Generate audio alert (beep sound)
        """
        import winsound
        try:
            winsound.Beep(1000, 500)  # frequency, duration
        except:
            pass  # Fallback for non-Windows systems
    
    @staticmethod
    def create_email_alert(recipient, class_name, confidence, timestamp):
        """
        Create email alert (requires email configuration)
        """
        subject = f"Exam Alert: {class_name} detected"
        body = f"""
        Suspicious activity detected:
        Type: {class_name}
        Confidence: {confidence:.2f}
        Time: {timestamp}
        
        Please review the evidence and take appropriate action.
        """
        return {'subject': subject, 'body': body, 'recipient': recipient}


# Cheating behavior definitions
CHEATING_BEHAVIORS = {
    'look_left': {'severity': 'high', 'description': 'Looking to the left (possible cheating)'},
    'look_right': {'severity': 'high', 'description': 'Looking to the right (possible cheating)'},
    'look_up': {'severity': 'high', 'description': 'Looking upward (possible cheating)'},
    'look_down': {'severity': 'medium', 'description': 'Looking downward (possible cheating)'},
    'mouth_open': {'severity': 'medium', 'description': 'Mouth open (possible communication)'},
    'see_left': {'severity': 'high', 'description': 'Viewing to the left'},
    'see_right': {'severity': 'high', 'description': 'Viewing to the right'},
    'see_down': {'severity': 'medium', 'description': 'Viewing downward'},
}
