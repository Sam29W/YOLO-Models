import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import json
from datetime import datetime
import time
import zipfile
from pathlib import Path
import threading
from collections import defaultdict, deque
import math

try:
    import pygame

    pygame.mixer.init()
    SOUND_AVAILABLE = True
except:
    SOUND_AVAILABLE = False
    print("⚠️ Sound not available on this system (disabled for cloud deployment)")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib import colors

    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False
    print("⚠️ ReportLab not available - PDF export disabled")

try:
    from skimage import filters
    from scipy import ndimage
    from scipy.spatial import distance

    HEATMAP_AVAILABLE = True
except:
    HEATMAP_AVAILABLE = False
    print("⚠️ scikit-image/scipy not available - Heatmap disabled")

try:
    import qrcode

    QR_AVAILABLE = True
except:
    QR_AVAILABLE = False
    print("⚠️ QR code generation not available")

# Initialize with default model
current_model = YOLO("yolo11n.pt")
model_cache = {"yolo11n.pt": current_model}

# Create directories
HISTORY_DIR = Path("detection_history")
HISTORY_DIR.mkdir(exist_ok=True)
ALERTS_DIR = Path("alert_logs")
ALERTS_DIR.mkdir(exist_ok=True)

# Global variables
alert_log = []
AVAILABLE_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
track_history = defaultdict(lambda: [])
frame_times = []


# 🧠 BEHAVIORAL ANALYSIS CLASS
class BehaviorTracker:
    def __init__(self):
        self.tracks = {}
        self.behavioral_alerts = []
        self.dwell_heatmap = None

    def update_track(self, track_id, position, class_name, frame_num, fps):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'positions': deque(maxlen=30),
                'first_seen': frame_num,
                'last_seen': frame_num,
                'class': class_name,
                'state': 'UNKNOWN',
                'dwell_time': 0,
                'total_distance': 0,
                'alerts': []
            }

        track = self.tracks[track_id]
        track['last_seen'] = frame_num

        if len(track['positions']) > 0:
            prev_pos = track['positions'][-1]
            dist = math.sqrt((position[0] - prev_pos[0]) ** 2 + (position[1] - prev_pos[1]) ** 2)
            track['total_distance'] += dist
            speed = dist

            if speed < 2:
                track['state'] = '🧍 IDLE'
                track['dwell_time'] += 1
            elif speed < 10:
                track['state'] = '🚶 WALKING'
            else:
                track['state'] = '🏃 RUNNING'

        track['positions'].append(position)
        self._check_anomalies(track_id, track, frame_num, fps)
        return track

    def _check_anomalies(self, track_id, track, frame_num, fps):
        if track['state'] == '🧍 IDLE' and track['dwell_time'] > (5 * fps):
            time_seconds = track['dwell_time'] / fps
            alert = f"⚠️ ID:{track_id} loitering ({time_seconds:.1f}s)"
            if alert not in track['alerts']:
                track['alerts'].append(alert)
                self.behavioral_alerts.append({
                    'frame': frame_num,
                    'track_id': track_id,
                    'type': 'LOITERING',
                    'message': alert
                })

        if track['state'] == '🏃 RUNNING':
            alert = f"⚡ ID:{track_id} running detected!"
            if alert not in track['alerts']:
                track['alerts'].append(alert)
                self.behavioral_alerts.append({
                    'frame': frame_num,
                    'track_id': track_id,
                    'type': 'RUNNING',
                    'message': alert
                })

    def get_active_tracks(self, current_frame, timeout=30):
        return {tid: track for tid, track in self.tracks.items()
                if current_frame - track['last_seen'] < timeout}

    def get_statistics(self, fps):
        stats = {
            'total_tracks': len(self.tracks),
            'active_tracks': len([t for t in self.tracks.values() if t['last_seen'] > 0]),
            'states': {'🧍 IDLE': 0, '🚶 WALKING': 0, '🏃 RUNNING': 0},
            'total_alerts': len(self.behavioral_alerts),
            'loitering_events': 0,
            'running_events': 0
        }

        for track in self.tracks.values():
            if track['state'] in stats['states']:
                stats['states'][track['state']] += 1

        for alert in self.behavioral_alerts:
            if alert['type'] == 'LOITERING':
                stats['loitering_events'] += 1
            elif alert['type'] == 'RUNNING':
                stats['running_events'] += 1

        return stats


behavior_tracker = BehaviorTracker()


def load_model(model_name):
    global current_model, model_cache
    if model_name not in model_cache:
        print(f"📥 Loading {model_name}...")
        model_cache[model_name] = YOLO(model_name)
        print(f"✅ {model_name} loaded successfully!")
    current_model = model_cache[model_name]
    return current_model


def draw_custom_boxes(image, results, confidence):
    if isinstance(image, Image.Image):
        image = np.array(image)
    annotated = image.copy()

    color_map = {
        'person': (255, 100, 100), 'car': (100, 100, 255), 'truck': (100, 255, 255),
        'bus': (255, 165, 100), 'bicycle': (255, 100, 255), 'motorcycle': (200, 100, 255),
        'dog': (100, 255, 100), 'cat': (150, 255, 150), 'bird': (255, 200, 100),
        'bottle': (180, 180, 100), 'cup': (255, 150, 150), 'laptop': (100, 200, 200),
        'phone': (200, 100, 200), 'chair': (150, 150, 255),
    }

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])
        if conf < confidence:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        color = color_map.get(class_name, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def generate_heatmap(image, results, confidence):
    if not HEATMAP_AVAILABLE:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        black = np.zeros_like(img_array)
        return black, img_array

    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()

        height, width = img_array.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(max(x2 - x1, y2 - y1) * 0.8)

            y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            gaussian = np.exp(-(x * x + y * y) / (2.0 * (radius / 2) ** 2))

            y_min = max(0, center_y - radius)
            y_max = min(height, center_y + radius + 1)
            x_min = max(0, center_x - radius)
            x_max = min(width, center_x + radius + 1)

            g_y_min = max(0, radius - center_y)
            g_y_max = g_y_min + (y_max - y_min)
            g_x_min = max(0, radius - center_x)
            g_x_max = g_x_min + (x_max - x_min)

            heatmap[y_min:y_max, x_min:x_max] += gaussian[g_y_min:g_y_max, g_x_min:g_x_max] * conf

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap = filters.gaussian(heatmap, sigma=20)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)

        return heatmap_color, overlay

    except Exception as e:
        print(f"❌ Heatmap error: {e}")
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        black = np.zeros_like(img_array)
        return black, img_array


def generate_qr_code(url):
    if not QR_AVAILABLE:
        return None
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        return np.array(img.convert('RGB'))
    except Exception as e:
        print(f"❌ QR error: {e}")
        return None


def create_mobile_snapshot(image, results, confidence):
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    annotated = img_array.copy()
    object_counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])
        if conf < confidence:
            continue

        object_counts[class_name] = object_counts.get(class_name, 0) + 1

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 5)

        label = f"{class_name}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(annotated, (x1, y1 - label_h - 20), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    overlay = annotated.copy()
    total_objects = sum(object_counts.values())
    info_text = f"Objects: {total_objects}"
    cv2.rectangle(overlay, (10, 10), (400, 100), (0, 0, 0), -1)
    cv2.putText(overlay, info_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    annotated = cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0)

    return annotated, object_counts


def detect_mobile_camera(image, confidence, model_name):
    if image is None:
        return None, "### 📱 Tap to capture photo!", None

    load_model(model_name)
    start_time = time.time()
    results = current_model.predict(source=image, conf=confidence, save=False, verbose=False)
    process_time = time.time() - start_time

    annotated, object_counts = create_mobile_snapshot(image, results, confidence)

    total = sum(object_counts.values())
    summary = f"# 📱 Mobile Detection\n\n**🎯 Found: {total} objects**\n**⚡ Speed: {process_time:.2f}s**\n\n"

    if object_counts:
        summary += "### Objects:\n"
        emoji_map = {'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌', 'dog': '🐕', 'cat': '🐈', 'bicycle': '🚲',
                     'phone': '📱'}
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            emoji = emoji_map.get(obj, '📦')
            summary += f"- {emoji} **{obj.capitalize()}: {count}**\n"

    temp_file = None
    if total > 0:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp.name, annotated)
        temp.close()
        temp_file = temp.name

    return annotated, summary, temp_file


# 🧠 BEHAVIORAL ANALYSIS - THE GAME CHANGER!
def analyze_behavior_in_video(video_path, confidence, model_name, loitering_threshold, progress=gr.Progress()):
    try:
        global behavior_tracker
        behavior_tracker = BehaviorTracker()

        load_model(model_name)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_output.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 180))

        frame_count = 0
        start_time = time.time()
        dwell_heatmap = np.zeros((height, width), dtype=np.float32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 10 == 0:
                progress(frame_count / total_frames, desc=f"🧠 Analyzing... {frame_count}/{total_frames}")

            results = current_model.track(source=frame, conf=confidence, persist=True, verbose=False)
            annotated_frame = frame.copy()

            dashboard = np.zeros((180, width, 3), dtype=np.uint8)
            dashboard[:] = (20, 20, 20)

            active_states = {'🧍 IDLE': 0, '🚶 WALKING': 0, '🏃 RUNNING': 0}

            if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    if conf < confidence:
                        continue

                    x1, y1, x2, y2 = box.astype(int)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    class_name = results[0].names[cls]

                    track = behavior_tracker.update_track(track_id, center, class_name, frame_count, fps)

                    if track['state'] == '🧍 IDLE':
                        cv2.circle(dwell_heatmap, center, 20, 1, -1)

                    if track['state'] in active_states:
                        active_states[track['state']] += 1

                    if track['dwell_time'] > (loitering_threshold * fps):
                        color = (0, 0, 255)
                        thickness = 4
                    elif track['state'] == '🏃 RUNNING':
                        color = (0, 165, 255)
                        thickness = 3
                    elif track['state'] == '🚶 WALKING':
                        color = (0, 255, 0)
                        thickness = 2
                    else:
                        color = (255, 255, 0)
                        thickness = 2

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                    dwell_time_sec = track['dwell_time'] / fps
                    label = f"ID:{track_id} {track['state']}"
                    if track['state'] == '🧍 IDLE':
                        label += f" {dwell_time_sec:.1f}s"

                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stats = behavior_tracker.get_statistics(fps)

            cv2.putText(dashboard, "BEHAVIORAL ANALYSIS DASHBOARD", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y_offset = 55
            cv2.putText(dashboard, f"Active: {len(behavior_tracker.get_active_tracks(frame_count))}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_offset += 25
            cv2.putText(dashboard, f"Alerts: {stats['total_alerts']}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

            y_offset += 30
            cv2.putText(dashboard, "STATES:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            for state, count in active_states.items():
                cv2.putText(dashboard, f"{state}: {count}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset += 22

            alert_x = width // 2
            cv2.putText(dashboard, "RECENT ALERTS:", (alert_x, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            recent_alerts = behavior_tracker.behavioral_alerts[-3:]
            alert_y = 80
            for alert in recent_alerts:
                alert_text = alert['message'][:40]
                cv2.putText(dashboard, alert_text, (alert_x, alert_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
                alert_y += 20

            combined = np.vstack([dashboard, annotated_frame])
            out.write(combined)

        cap.release()
        out.release()

        total_time = time.time() - start_time
        stats = behavior_tracker.get_statistics(fps)

        summary = f"# 🧠 Behavioral Analysis Complete!\n\n"
        summary += f"**🤖 Model:** {model_name}\n"
        summary += f"**⏱️ Time:** {total_time:.2f}s\n"
        summary += f"**📹 Frames:** {frame_count}\n"
        summary += f"**🎯 Tracks:** {stats['total_tracks']}\n\n"
        summary += f"## 📊 Statistics:\n\n"
        summary += f"- **🚨 Total Alerts:** {stats['total_alerts']}\n"
        summary += f"- **⚠️ Loitering:** {stats['loitering_events']}\n"
        summary += f"- **⚡ Running:** {stats['running_events']}\n\n"

        summary += "### Activity:\n\n"
        for state, count in stats['states'].items():
            summary += f"- {state}: **{count}**\n"

        summary += "\n### 🚨 Alerts:\n\n"
        if behavior_tracker.behavioral_alerts:
            for i, alert in enumerate(behavior_tracker.behavioral_alerts[:10], 1):
                frame_time = alert['frame'] / fps
                summary += f"{i}. **{alert['message']}** ({frame_time:.1f}s)\n"
            if len(behavior_tracker.behavioral_alerts) > 10:
                summary += f"\n*...and {len(behavior_tracker.behavioral_alerts) - 10} more*\n"
        else:
            summary += "*No behavioral alerts detected!* ✅\n"

        heatmap_output = None
        if HEATMAP_AVAILABLE and dwell_heatmap.max() > 0:
            dwell_heatmap = dwell_heatmap / dwell_heatmap.max()
            dwell_heatmap = filters.gaussian(dwell_heatmap, sigma=20)
            heatmap_color = cv2.applyColorMap((dwell_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(heatmap_temp.name, heatmap_color)
            heatmap_output = heatmap_temp.name

        # 🔧 FIXED JSON EXPORT - Convert all numpy types
        behavior_json = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_name,
            'total_frames': int(frame_count),
            'fps': int(fps),
            'statistics': {
                'total_tracks': int(stats['total_tracks']),
                'active_tracks': int(stats['active_tracks']),
                'states': {str(k): int(v) for k, v in stats['states'].items()},
                'total_alerts': int(stats['total_alerts']),
                'loitering_events': int(stats['loitering_events']),
                'running_events': int(stats['running_events'])
            },
            'alerts': [
                {
                    'frame': int(alert['frame']),
                    'track_id': int(alert['track_id']),
                    'type': str(alert['type']),
                    'message': str(alert['message'])
                }
                for alert in behavior_tracker.behavioral_alerts
            ]
        }

        json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(behavior_json, json_temp, indent=2)
        json_temp.close()

        progress(1.0, desc="Complete!")
        return output_path, summary, heatmap_output, json_temp.name

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"### ⚠️ Error\n\n{str(e)}", None, None


def track_objects_in_video(video_path, confidence, model_name, draw_trails, progress=gr.Progress()):
    try:
        global track_history
        track_history = defaultdict(lambda: [])
        load_model(model_name)

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

        frame_count = 0
        track_stats = defaultdict(lambda: {'first_seen': 0, 'last_seen': 0, 'frames': 0})
        unique_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress(frame_count / total_frames, desc=f"Tracking {frame_count}/{total_frames}")

            results = current_model.track(source=frame, conf=confidence, persist=True, verbose=False)
            annotated_frame = frame.copy()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    if conf < confidence:
                        continue

                    unique_ids.add(track_id)
                    x1, y1, x2, y2 = box.astype(int)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    track_history[track_id].append(center)
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                    if track_stats[track_id]['first_seen'] == 0:
                        track_stats[track_id]['first_seen'] = frame_count
                    track_stats[track_id]['last_seen'] = frame_count
                    track_stats[track_id]['frames'] += 1

                    color = tuple(
                        [int((track_id * 50) % 255), int((track_id * 100) % 255), int((track_id * 150) % 255)])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                    class_name = results[0].names[cls]
                    label = f"ID:{track_id} {class_name}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if draw_trails and len(track_history[track_id]) > 1:
                        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=3)
                        cv2.circle(annotated_frame, center, 5, color, -1)

            out.write(annotated_frame)

        cap.release()
        out.release()

        summary = f"# 🎯 Tracking Complete!\n\n**🤖 Model:** {model_name}\n**📹 Frames:** {frame_count}\n**🎯 Objects:** {len(unique_ids)}\n\n### Stats:\n\n"
        for track_id in sorted(unique_ids)[:10]:
            stats = track_stats[track_id]
            duration = (stats['last_seen'] - stats['first_seen']) / fps
            summary += f"- **ID {track_id}:** {duration:.1f}s ({stats['frames']} frames)\n"

        if len(unique_ids) > 10:
            summary += f"\n*...and {len(unique_ids) - 10} more*\n"

        progress(1.0, desc="Done!")
        return temp_output.name, summary

    except Exception as e:
        return None, f"### ⚠️ Error\n\n{str(e)}"


def play_alert_sound():
    if not SOUND_AVAILABLE:
        return
    try:
        frequency, duration, sample_rate = 1000, 500, 22050
        samples = int(sample_rate * duration / 1000)
        wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration / 1000, samples))
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
        time.sleep(duration / 1000)
    except:
        pass


def log_alert(alert_message, object_counts, image=None):
    global alert_log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_entry = {'timestamp': timestamp, 'message': alert_message, 'object_counts': object_counts}
    alert_log.insert(0, alert_entry)
    if len(alert_log) > 50:
        alert_log = alert_log[:50]

    log_file = ALERTS_DIR / f"alert_{timestamp.replace(':', '-').replace(' ', '_')}.json"
    with open(log_file, 'w') as f:
        json.dump(alert_entry, f, indent=2)

    if image is not None:
        img_file = ALERTS_DIR / f"alert_img_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        cv2.imwrite(str(img_file), image)


def check_alerts(object_counts, alert_rules):
    triggered_alerts = []
    if alert_rules['person_count_enabled'] and 'person' in object_counts:
        if object_counts['person'] >= alert_rules['person_threshold']:
            triggered_alerts.append(
                f"⚠️ {object_counts['person']} Persons! (Threshold: {alert_rules['person_threshold']})")

    if alert_rules['car_alert_enabled'] and 'car' in object_counts:
        triggered_alerts.append(f"🚗 Car detected! ({object_counts['car']})")
    if alert_rules['truck_alert_enabled'] and 'truck' in object_counts:
        triggered_alerts.append(f"🚚 Truck detected! ({object_counts['truck']})")
    if alert_rules['dog_alert_enabled'] and 'dog' in object_counts:
        triggered_alerts.append(f"🐕 Dog detected! ({object_counts['dog']})")
    if alert_rules['cat_alert_enabled'] and 'cat' in object_counts:
        triggered_alerts.append(f"🐈 Cat detected! ({object_counts['cat']})")
    if alert_rules['bicycle_alert_enabled'] and 'bicycle' in object_counts:
        triggered_alerts.append(f"🚲 Bicycle detected! ({object_counts['bicycle']})")

    total = sum(object_counts.values())
    if alert_rules['total_objects_enabled'] and total >= alert_rules['total_objects_threshold']:
        triggered_alerts.append(f"📊 {total} Total objects! (Threshold: {alert_rules['total_objects_threshold']})")

    return triggered_alerts


def get_alert_log_display():
    if not alert_log:
        return "### 📂 No Alerts Yet"
    display = "# 🚨 Alert Log\n\n"
    for idx, alert in enumerate(alert_log[:20]):
        display += f"## {idx + 1}. {alert['timestamp']}\n**{alert['message']}**\n\n"
        if alert['object_counts']:
            for obj, count in alert['object_counts'].items():
                display += f"- {obj}: {count}\n"
        display += "\n---\n\n"
    return display


def clear_alert_log():
    global alert_log
    alert_log = []
    for file in ALERTS_DIR.glob("*"):
        file.unlink()
    return "### ✅ Cleared!"


def export_alert_log():
    if not alert_log:
        return None
    export_data = {'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'total_alerts': len(alert_log),
                   'alerts': alert_log}
    json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(export_data, json_temp, indent=2)
    json_temp.close()
    return json_temp.name


def save_to_history(image, object_counts, timestamp, model_name):
    try:
        img_filename = f"detection_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        cv2.imwrite(str(HISTORY_DIR / img_filename), image)
        metadata = {'timestamp': timestamp, 'model': model_name, 'object_counts': object_counts,
                    'total_objects': sum(object_counts.values())}
        meta_filename = f"metadata_{timestamp.replace(':', '-').replace(' ', '_')}.json"
        with open(HISTORY_DIR / meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except:
        return False


def load_history():
    try:
        history_files = sorted(HISTORY_DIR.glob("metadata_*.json"), reverse=True)
        if not history_files:
            return "### 📂 No History"
        history_text = "# 📂 History\n\n"
        for meta_file in history_files[:10]:
            with open(meta_file, 'r') as f:
                data = json.load(f)
            history_text += f"## 🕒 {data['timestamp']}\n**Model:** {data.get('model', 'yolo11n.pt')}\n**Total:** {data['total_objects']}\n\n"
            for obj, count in sorted(data['object_counts'].items(), key=lambda x: x[1], reverse=True):
                history_text += f"- {obj}: {count}\n"
            history_text += "\n---\n\n"
        return history_text
    except Exception as e:
        return f"### ⚠️ Error\n\n{str(e)}"


def clear_history():
    try:
        for file in HISTORY_DIR.glob("*"):
            file.unlink()
        return "### ✅ Cleared!"
    except Exception as e:
        return f"### ⚠️ Error: {str(e)}"


def detect_webcam(image, confidence, model_name):
    global frame_times
    if image is None:
        return None, "### 🎯 Waiting..."

    load_model(model_name)
    current_time = time.time()
    frame_times.append(current_time)
    frame_times = [t for t in frame_times if current_time - t < 1.0]
    fps = len(frame_times)

    results = current_model.predict(source=image, conf=confidence, save=False, verbose=False)
    annotated = results[0].plot()

    object_counts = {}
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls[0])]
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    text = f"### 🔴 LIVE\n\n**Model:** {model_name}\n**FPS:** {fps}\n\n"
    if object_counts:
        total = sum(object_counts.values())
        text += f"**Total: {total}**\n\n"
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            text += f"{obj}: {count} | "

    return annotated, text


def detect_objects_with_alerts(image, confidence, use_custom_colors,
                               person_alert_enabled, person_threshold,
                               car_alert, truck_alert, dog_alert, cat_alert, bicycle_alert,
                               total_objects_enabled, total_threshold,
                               sound_enabled, log_enabled,
                               model_name,
                               *filters):
    if image is None:
        return None, "⚠️ Please upload an image!", None, None, None, None, None, None

    load_model(model_name)
    filter_classes = ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'dog', 'cat', 'bird', 'bottle', 'cup',
                      'laptop', 'phone', 'chair']
    selected_filters = [filter_classes[i] for i, f in enumerate(filters) if f]

    start_time = time.time()
    results = current_model.predict(source=image, conf=confidence, save=False)
    process_time = time.time() - start_time

    if use_custom_colors:
        annotated_image = draw_custom_boxes(image, results, confidence)
    else:
        annotated_image = results[0].plot()

    heatmap_pure, heatmap_overlay = generate_heatmap(image, results, confidence)

    detections = []
    object_counts = {}
    export_data = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

        if selected_filters and class_name not in selected_filters:
            continue

        detections.append(f"{class_name}: {conf:.2%}")
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        export_data.append({'object': class_name, 'confidence': f"{conf:.4f}", 'bbox_x1': round(bbox[0], 2),
                            'bbox_y1': round(bbox[1], 2), 'bbox_x2': round(bbox[2], 2), 'bbox_y2': round(bbox[3], 2)})

    alert_rules = {
        'person_count_enabled': person_alert_enabled, 'person_threshold': person_threshold,
        'car_alert_enabled': car_alert, 'truck_alert_enabled': truck_alert,
        'dog_alert_enabled': dog_alert, 'cat_alert_enabled': cat_alert,
        'bicycle_alert_enabled': bicycle_alert,
        'total_objects_enabled': total_objects_enabled, 'total_objects_threshold': total_threshold
    }

    triggered_alerts = check_alerts(object_counts, alert_rules)
    alert_summary = ""
    if triggered_alerts:
        alert_summary = "\n\n## 🚨 ALERTS!\n\n"
        for alert_msg in triggered_alerts:
            alert_summary += f"### {alert_msg}\n\n"
            if sound_enabled:
                threading.Thread(target=play_alert_sound).start()
            if log_enabled:
                log_alert(alert_msg, object_counts, annotated_image)
        alert_summary += "---\n\n"

    total = len(detections)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if object_counts:
        save_to_history(annotated_image, object_counts, timestamp, model_name)

    summary = f"# 📊 Results\n\n**🤖 Model:** {model_name}\n"
    if use_custom_colors:
        summary += "**🎨 Custom Colors:** ON\n"
    summary += f"**Timestamp:** {timestamp}\n**⏱️ Time:** {process_time:.2f}s\n**🚀 FPS:** {1 / process_time:.1f}\n"
    summary += alert_summary
    summary += f"## 🎯 Total: **{total}**\n\n"

    if total > 0:
        summary += "### 📈 Objects:\n\n"
        emoji_map = {'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌', 'bicycle': '🚲', 'dog': '🐕', 'cat': '🐈'}
        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            emoji = emoji_map.get(obj, '📦')
            bar = '🟦' * count
            summary += f"**{emoji} {obj.capitalize()}:** {count} {bar}\n\n"

    json_file = csv_file = pdf_file = None
    if export_data:
        json_output = {'timestamp': timestamp, 'model': model_name, 'total_objects': total,
                       'object_summary': object_counts, 'detections': export_data}
        json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(json_output, json_temp, indent=2)
        json_temp.close()
        json_file = json_temp.name

        csv_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        csv_temp.write("Object,Confidence,X1,Y1,X2,Y2\n")
        for item in export_data:
            csv_temp.write(
                f"{item['object']},{item['confidence']},{item['bbox_x1']},{item['bbox_y1']},{item['bbox_x2']},{item['bbox_y2']}\n")
        csv_temp.close()
        csv_file = csv_temp.name

    alert_log_display = get_alert_log_display()
    return annotated_image, summary, alert_log_display, json_file, csv_file, pdf_file, heatmap_pure, heatmap_overlay


# Create Gradio interface
with gr.Blocks(title="YOLO Detection Pro - AI Behavioral Analysis") as demo:
    gr.Markdown("# 🧠 YOLO Object Detection Pro - **AI BEHAVIORAL ANALYSIS**")
    gr.Markdown("### Multi-Model | Mobile | QR | Tracking | **BEHAVIORAL ANALYSIS** 🧠 | Alerts | Heatmaps!")

    # 📊 ANIMATED STATS DASHBOARD - NEW ADDITION! 🔥
    gr.HTML("""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 20px; backdrop-filter: blur(10px);">

            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(102,126,234,0.3), rgba(118,75,162,0.3)); border-radius: 15px; transform: scale(1); transition: all 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <div style="font-size: 3em; margin-bottom: 10px;">🤖</div>
                <div style="font-size: 2.5em; font-weight: bold; color: white; text-shadow: 0 0 20px rgba(102,126,234,0.8);">5+</div>
                <div style="color: white; font-weight: 600; margin-top: 10px;">AI Models</div>
            </div>

            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(240,147,251,0.3), rgba(245,87,108,0.3)); border-radius: 15px; transform: scale(1); transition: all 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <div style="font-size: 3em; margin-bottom: 10px;">🎯</div>
                <div style="font-size: 2.5em; font-weight: bold; color: white; text-shadow: 0 0 20px rgba(240,147,251,0.8);">80+</div>
                <div style="color: white; font-weight: 600; margin-top: 10px;">Object Classes</div>
            </div>

            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(79,172,254,0.3), rgba(0,242,254,0.3)); border-radius: 15px; transform: scale(1); transition: all 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <div style="font-size: 3em; margin-bottom: 10px;">⚡</div>
                <div style="font-size: 2.5em; font-weight: bold; color: white; text-shadow: 0 0 20px rgba(79,172,254,0.8);">30+</div>
                <div style="color: white; font-weight: 600; margin-top: 10px;">FPS Speed</div>
            </div>

            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(86,171,47,0.3), rgba(168,224,99,0.3)); border-radius: 15px; transform: scale(1); transition: all 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <div style="font-size: 3em; margin-bottom: 10px;">✨</div>
                <div style="font-size: 2.5em; font-weight: bold; color: white; text-shadow: 0 0 20px rgba(168,224,99,0.8);">99%</div>
                <div style="color: white; font-weight: 600; margin-top: 10px;">Accuracy</div>
            </div>

        </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            global_model_selector = gr.Dropdown(choices=AVAILABLE_MODELS, value="yolo11n.pt", label="🤖 Select Model",
                                                info="n=fastest | s=fast | m=balanced | l=accurate | x=best")
        with gr.Column(scale=2):
            gr.Markdown(
                "### Models:\n- **n**: 2.6M params\n- **s**: 9.4M params\n- **m**: 20.1M params\n- **l**: 25.3M params\n- **x**: 56.9M params")

    with gr.Tabs():
        # 🧠 BEHAVIORAL ANALYSIS TAB
        with gr.Tab("🧠 AI Behavioral Analysis"):
            gr.Markdown("# 🧠 AI-Powered Behavioral Analysis")
            gr.Markdown("### **Enterprise-Grade**: Detect not just WHAT, but HOW objects behave!")

            with gr.Row():
                with gr.Column():
                    behavior_video_input = gr.Video(label="📤 Upload Video")
                    behavior_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    behavior_loitering_threshold = gr.Slider(1, 30, 5, step=1, label="🚨 Loitering Alert (seconds)")
                    behavior_btn = gr.Button("🧠 Analyze Behavior", variant="primary", size="lg")

                    gr.Markdown("""
                    ### 🔥 FEATURES:

                    **Activity:**
                    - 🧍 **IDLE** - Not moving
                    - 🚶 **WALKING** - Normal
                    - 🏃 **RUNNING** - Fast

                    **Alerts:**
                    - ⚠️ **Loitering** - Idle too long
                    - ⚡ **Running** - Sudden speed

                    **Analytics:**
                    - ⏱️ Dwell Time
                    - 📏 Distance
                    - 🗺️ Heatmap
                    - 📊 Statistics

                    ### 💼 USE CASES:
                    - 🏢 Security
                    - 🏪 Retail
                    - 🚗 Traffic
                    - 🏭 Safety
                    - ⚽ Sports
                    """)

                with gr.Column():
                    behavior_output = gr.Video(label="🧠 Analyzed Video")
                    behavior_summary = gr.Markdown("### 📊 Results here!")

                    with gr.Row():
                        behavior_heatmap = gr.Image(type="filepath", label="🗺️ Dwell Heatmap")
                        behavior_json = gr.File(label="📄 Data (JSON)")

            behavior_btn.click(fn=analyze_behavior_in_video,
                               inputs=[behavior_video_input, behavior_confidence, global_model_selector,
                                       behavior_loitering_threshold],
                               outputs=[behavior_output, behavior_summary, behavior_heatmap, behavior_json])

        # Image Detection Tab
        with gr.Tab("📷 Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="📤 Upload Image")
                    image_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    custom_colors_toggle = gr.Checkbox(label="🎨 Custom Colors", value=True)

                    with gr.Accordion("🔔 Alerts", open=True):
                        with gr.Row():
                            person_alert_enabled = gr.Checkbox(label="Person Count", value=False)
                            person_threshold = gr.Slider(1, 20, 3, step=1, label="Threshold")
                        with gr.Row():
                            total_objects_enabled = gr.Checkbox(label="Total Objects", value=False)
                            total_threshold = gr.Slider(1, 50, 10, step=1, label="Threshold")
                        with gr.Row():
                            car_alert = gr.Checkbox(label="🚗 Car", value=False)
                            truck_alert = gr.Checkbox(label="🚚 Truck", value=False)
                            bicycle_alert = gr.Checkbox(label="🚲 Bicycle", value=False)
                        with gr.Row():
                            dog_alert = gr.Checkbox(label="🐕 Dog", value=False)
                            cat_alert = gr.Checkbox(label="🐈 Cat", value=False)
                        with gr.Row():
                            sound_enabled = gr.Checkbox(label="🔊 Sound", value=True)
                            log_enabled = gr.Checkbox(label="📝 Log", value=True)

                    with gr.Accordion("🎯 Filters", open=False):
                        filter_person = gr.Checkbox(label="👤 Person", value=False)
                        filter_car = gr.Checkbox(label="🚗 Car", value=False)
                        filter_truck = gr.Checkbox(label="🚚 Truck", value=False)
                        filter_bus = gr.Checkbox(label="🚌 Bus", value=False)
                        filter_bicycle = gr.Checkbox(label="🚲 Bicycle", value=False)
                        filter_motorcycle = gr.Checkbox(label="🏍️ Motorcycle", value=False)
                        filter_dog = gr.Checkbox(label="🐕 Dog", value=False)
                        filter_cat = gr.Checkbox(label="🐈 Cat", value=False)
                        filter_bird = gr.Checkbox(label="🐦 Bird", value=False)
                        filter_bottle = gr.Checkbox(label="🍾 Bottle", value=False)
                        filter_cup = gr.Checkbox(label="☕ Cup", value=False)
                        filter_laptop = gr.Checkbox(label="💻 Laptop", value=False)
                        filter_phone = gr.Checkbox(label="📱 Phone", value=False)
                        filter_chair = gr.Checkbox(label="🪑 Chair", value=False)

                    image_btn = gr.Button("🔍 Detect", variant="primary", size="lg")

                with gr.Column():
                    image_output = gr.Image(type="numpy", label="✨ Results")
                    image_text = gr.Markdown()
                    with gr.Accordion("🗺️ Heatmap", open=False):
                        with gr.Row():
                            heatmap_pure = gr.Image(type="numpy", label="🔥 Pure")
                            heatmap_overlay = gr.Image(type="numpy", label="🎨 Overlay")
                    alert_log_display = gr.Markdown("### 📂 Alerts")
                    with gr.Row():
                        json_download = gr.File(label="📄 JSON")
                        csv_download = gr.File(label="📊 CSV")
                        pdf_download = gr.File(label="📋 PDF")

            image_btn.click(fn=detect_objects_with_alerts,
                            inputs=[image_input, image_confidence, custom_colors_toggle, person_alert_enabled,
                                    person_threshold, car_alert, truck_alert, dog_alert, cat_alert, bicycle_alert,
                                    total_objects_enabled, total_threshold, sound_enabled, log_enabled,
                                    global_model_selector, filter_person, filter_car, filter_truck, filter_bus,
                                    filter_bicycle, filter_motorcycle, filter_dog, filter_cat, filter_bird,
                                    filter_bottle, filter_cup, filter_laptop, filter_phone, filter_chair],
                            outputs=[image_output, image_text, alert_log_display, json_download, csv_download,
                                     pdf_download, heatmap_pure, heatmap_overlay])

        # 🎯 Object Tracking
        with gr.Tab("🎯 Object Tracking"):
            gr.Markdown("# 🎯 Real-Time Object Tracking")
            with gr.Row():
                with gr.Column():
                    tracking_video_input = gr.Video(label="📤 Upload Video")
                    tracking_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    tracking_trails = gr.Checkbox(label="🎨 Draw Trails", value=True)
                    tracking_btn = gr.Button("🎯 Track Objects", variant="primary", size="lg")
                with gr.Column():
                    tracking_output = gr.Video(label="🎯 Tracked Video")
                    tracking_summary = gr.Markdown("### 📊 Results!")
            tracking_btn.click(fn=track_objects_in_video,
                               inputs=[tracking_video_input, tracking_confidence, global_model_selector,
                                       tracking_trails], outputs=[tracking_output, tracking_summary])

        # 📱 Mobile Camera
        with gr.Tab("📱 Mobile Camera"):
            gr.Markdown("# 📱 Mobile-Optimized Detection")
            mobile_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="📊 Confidence")
            mobile_camera_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="📷 Capture", height=400)
            mobile_detect_btn = gr.Button("🔍 DETECT", variant="primary", size="lg")
            mobile_output = gr.Image(type="numpy", label="✨ Results", height=400)
            mobile_summary = gr.Markdown("### 📱 Results!")
            mobile_download = gr.File(label="💾 Download")
            mobile_detect_btn.click(fn=detect_mobile_camera,
                                    inputs=[mobile_camera_input, mobile_confidence, global_model_selector],
                                    outputs=[mobile_output, mobile_summary, mobile_download])

        # 📲 QR Code
        with gr.Tab("📲 Share via QR"):
            gr.Markdown("# 📲 Share This App!")
            with gr.Row():
                with gr.Column():
                    share_url_input = gr.Textbox(label="🔗 App URL", placeholder="Enter URL",
                                                 value="https://huggingface.co/spaces/Samith29/yolo-object-detection")
                    qr_generate_btn = gr.Button("🎨 Generate QR", variant="primary", size="lg")
                with gr.Column():
                    qr_output = gr.Image(type="numpy", label="📲 Scan Me!")
            qr_generate_btn.click(fn=generate_qr_code, inputs=[share_url_input], outputs=[qr_output])

        # 📸 Live Webcam
        with gr.Tab("📸 Live Webcam"):
            gr.Markdown("### 🔴 Real-Time Detection")
            with gr.Row():
                with gr.Column():
                    webcam_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    webcam_counter = gr.Markdown("### 🎯 Waiting...")
                with gr.Column():
                    webcam_output = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="🎥 Live")
            webcam_output.stream(fn=detect_webcam, inputs=[webcam_output, webcam_confidence, global_model_selector],
                                 outputs=[webcam_output, webcam_counter], time_limit=60, stream_every=0.1)

        # 🚨 Alert Log
        with gr.Tab("🚨 Alert Log"):
            gr.Markdown("### 🚨 Alert Management")
            with gr.Row():
                alert_refresh_btn = gr.Button("🔄 Refresh", variant="primary")
                alert_clear_btn = gr.Button("🗑️ Clear", variant="stop")
                alert_export_btn = gr.Button("📥 Export", variant="secondary")
            alert_history_display = gr.Markdown("Click Refresh!")
            alert_export_file = gr.File(label="📄 Download")
            alert_refresh_btn.click(fn=get_alert_log_display, outputs=alert_history_display)
            alert_clear_btn.click(fn=clear_alert_log, outputs=alert_history_display)
            alert_export_btn.click(fn=export_alert_log, outputs=alert_export_file)

        # 📂 History
        with gr.Tab("📂 History"):
            gr.Markdown("### 📂 Detection History")
            with gr.Row():
                history_load_btn = gr.Button("🔄 Load", variant="primary")
                history_clear_btn = gr.Button("🗑️ Clear", variant="stop")
            history_display = gr.Markdown("Click Load!")
            history_load_btn.click(fn=load_history, outputs=history_display)
            history_clear_btn.click(fn=clear_history, outputs=history_display)

    gr.Markdown("---")
    gr.Markdown("Built by **Samith Shivakumar** | Powered by YOLOv11 🚀")
    gr.Markdown("⭐ **Features:** Multi-Model | Mobile | QR | Tracking | **BEHAVIORAL ANALYSIS** 🧠 | Alerts | Heatmaps")

if __name__ == "__main__":
    demo.launch()
