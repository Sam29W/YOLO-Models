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

    HEATMAP_AVAILABLE = True
except:
    HEATMAP_AVAILABLE = False
    print("⚠️ scikit-image/scipy not available - Heatmap disabled")

# Initialize with default model
current_model = YOLO("yolo11n.pt")
model_cache = {"yolo11n.pt": current_model}

# Create history directory
HISTORY_DIR = Path("detection_history")
HISTORY_DIR.mkdir(exist_ok=True)

# Create alerts directory
ALERTS_DIR = Path("alert_logs")
ALERTS_DIR.mkdir(exist_ok=True)

# Global alert log
alert_log = []

# Available models
AVAILABLE_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]


def load_model(model_name):
    """
    Load or retrieve cached YOLO model
    """
    global current_model, model_cache

    if model_name not in model_cache:
        print(f"📥 Loading {model_name}...")
        model_cache[model_name] = YOLO(model_name)
        print(f"✅ {model_name} loaded successfully!")

    current_model = model_cache[model_name]
    return current_model


def draw_custom_boxes(image, results, confidence):
    """
    Draw custom colored bounding boxes on image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    annotated = image.copy()

    color_map = {
        'person': (255, 100, 100),
        'car': (100, 100, 255),
        'truck': (100, 255, 255),
        'bus': (255, 165, 100),
        'bicycle': (255, 100, 255),
        'motorcycle': (200, 100, 255),
        'dog': (100, 255, 100),
        'cat': (150, 255, 150),
        'bird': (255, 200, 100),
        'bottle': (180, 180, 100),
        'cup': (255, 150, 150),
        'laptop': (100, 200, 200),
        'phone': (200, 100, 200),
        'chair': (150, 150, 255),
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
    """
    Generate detection heatmap showing object density
    """
    if not HEATMAP_AVAILABLE:
        print("⚠️ Heatmap generation not available")
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

        # Create empty heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)

        # Add Gaussian blobs for each detection
        for box in results[0].boxes:
            conf = float(box.conf[0])

            if conf < confidence:
                continue

            # Get bounding box center
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Calculate radius based on box size
            box_width = x2 - x1
            box_height = y2 - y1
            radius = int(max(box_width, box_height) * 0.8)

            # Create meshgrid for Gaussian
            y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]

            # Gaussian formula
            gaussian = np.exp(-(x * x + y * y) / (2.0 * (radius / 2) ** 2))

            # Add to heatmap (with bounds checking)
            y_min = max(0, center_y - radius)
            y_max = min(height, center_y + radius + 1)
            x_min = max(0, center_x - radius)
            x_max = min(width, center_x + radius + 1)

            g_y_min = max(0, radius - center_y)
            g_y_max = g_y_min + (y_max - y_min)
            g_x_min = max(0, radius - center_x)
            g_x_max = g_x_min + (x_max - x_min)

            heatmap[y_min:y_max, x_min:x_max] += gaussian[g_y_min:g_y_max, g_x_min:g_x_max] * conf

        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Apply Gaussian blur for smoother visualization
        heatmap = filters.gaussian(heatmap, sigma=20)

        # Normalize again after blur
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Convert to color heatmap (blue -> green -> yellow -> red)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Create overlay
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)

        return heatmap_color, overlay

    except Exception as e:
        print(f"❌ Heatmap generation error: {e}")
        # Return black images as fallback
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        black = np.zeros_like(img_array)
        return black, img_array


def generate_video_heatmap(video_path, confidence, model_name, progress=gr.Progress()):
    """
    Generate aggregate heatmap for video showing detection hotspots
    """
    if not HEATMAP_AVAILABLE:
        print("⚠️ Heatmap generation not available")
        return None, None

    try:
        load_model(model_name)

        cap = cv2.VideoCapture(video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize aggregate heatmap
        aggregate_heatmap = np.zeros((height, width), dtype=np.float32)

        frame_count = 0
        sample_frame = None

        # Process every Nth frame for performance
        frame_skip = max(1, total_frames // 100)  # Process max 100 frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue

            progress(frame_count / total_frames, desc=f"Generating heatmap... {frame_count}/{total_frames}")

            # Save one frame for display
            if sample_frame is None:
                sample_frame = frame.copy()

            # Run detection
            results = current_model.predict(source=frame, conf=confidence, save=False, verbose=False)

            # Add detections to aggregate heatmap
            for box in results[0].boxes:
                conf = float(box.conf[0])

                if conf < confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                box_width = x2 - x1
                box_height = y2 - y1
                radius = int(max(box_width, box_height) * 0.8)

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

                aggregate_heatmap[y_min:y_max, x_min:x_max] += gaussian[g_y_min:g_y_max, g_x_min:g_x_max]

        cap.release()

        # Normalize
        if aggregate_heatmap.max() > 0:
            aggregate_heatmap = aggregate_heatmap / aggregate_heatmap.max()

        # Apply Gaussian blur
        aggregate_heatmap = filters.gaussian(aggregate_heatmap, sigma=25)

        if aggregate_heatmap.max() > 0:
            aggregate_heatmap = aggregate_heatmap / aggregate_heatmap.max()

        # Convert to color
        heatmap_color = cv2.applyColorMap((aggregate_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Create overlay if we have a sample frame
        if sample_frame is not None:
            overlay = cv2.addWeighted(sample_frame, 0.6, heatmap_color, 0.4, 0)
        else:
            overlay = heatmap_color

        progress(1.0, desc="Heatmap complete!")

        return heatmap_color, overlay

    except Exception as e:
        print(f"❌ Video heatmap error: {e}")
        return None, None


def generate_pdf_report(annotated_image, export_data, object_counts, metadata):
    """
    Generate professional PDF report with detection results
    """
    if not PDF_AVAILABLE:
        print("⚠️ PDF generation not available")
        return None

    try:
        # Create temp PDF file
        pdf_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = pdf_temp.name
        pdf_temp.close()

        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                                rightMargin=0.75 * inch, leftMargin=0.75 * inch,
                                topMargin=0.75 * inch, bottomMargin=0.75 * inch)

        # Container for PDF elements
        elements = []

        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#A23B72'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        normal_style = styles['Normal']

        # COVER PAGE
        elements.append(Spacer(1, 1.5 * inch))
        elements.append(Paragraph("🎯 YOLO Object Detection Report", title_style))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(f"<b>Generated:</b> {metadata['timestamp']}", normal_style))
        elements.append(Paragraph(f"<b>Model:</b> {metadata['model']}", normal_style))
        elements.append(Paragraph(f"<b>Confidence Threshold:</b> {metadata['confidence']}", normal_style))
        elements.append(Spacer(1, 0.5 * inch))

        # Summary box
        summary_data = [
            ['Total Objects', str(metadata['total_objects'])],
            ['Processing Time', f"{metadata['processing_time']:.2f}s"],
            ['FPS', f"{metadata['fps']:.1f}"],
            ['Unique Classes', str(len(object_counts))]
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 2 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#F18F01')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))

        elements.append(summary_table)
        elements.append(PageBreak())

        # DETECTION RESULTS PAGE
        elements.append(Paragraph("Detection Results", heading_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Save annotated image
        img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(img_temp.name, annotated_image)
        img_temp.close()

        # Add image to PDF (resize if needed)
        img = RLImage(img_temp.name, width=6 * inch, height=4 * inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3 * inch))

        # Object counts
        elements.append(Paragraph("Object Summary", heading_style))
        elements.append(Spacer(1, 0.1 * inch))

        if object_counts:
            count_data = [['Object Type', 'Count', 'Percentage']]
            total = sum(object_counts.values())

            for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total) * 100
                count_data.append([obj.capitalize(), str(count), f"{percentage:.1f}%"])

            count_table = Table(count_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
            count_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E8F4F8')])
            ]))

            elements.append(count_table)

        elements.append(PageBreak())

        # DETAILED DETECTIONS PAGE
        elements.append(Paragraph("Detailed Detections", heading_style))
        elements.append(Spacer(1, 0.2 * inch))

        if export_data:
            # Limit to first 30 detections for PDF
            detail_data = [['#', 'Object', 'Confidence', 'BBox (x1,y1,x2,y2)']]

            for idx, item in enumerate(export_data[:30], 1):
                bbox_str = f"({item['bbox_x1']}, {item['bbox_y1']}, {item['bbox_x2']}, {item['bbox_y2']})"
                detail_data.append([
                    str(idx),
                    item['object'].capitalize(),
                    item['confidence'],
                    bbox_str
                ])

            detail_table = Table(detail_data, colWidths=[0.5 * inch, 1.5 * inch, 1.2 * inch, 3.3 * inch])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FCE0EF')])
            ]))

            elements.append(detail_table)

            if len(export_data) > 30:
                elements.append(Spacer(1, 0.2 * inch))
                elements.append(Paragraph(
                    f"<i>Showing first 30 of {len(export_data)} total detections. See JSON/CSV for complete data.</i>",
                    normal_style))

        # FOOTER
        elements.append(Spacer(1, 0.5 * inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Paragraph("─" * 80, footer_style))
        elements.append(Paragraph("Generated by <b>YOLO Object Detection Pro</b>", footer_style))
        elements.append(Paragraph("Built by <b>Samith Shivakumar</b> | Powered by YOLOv11", footer_style))
        elements.append(Paragraph("GitHub: <link href='https://github.com/Sam29W'>Sam29W</link>", footer_style))

        # Build PDF
        doc.build(elements)

        # Clean up temp image
        os.unlink(img_temp.name)

        print(f"✅ PDF report generated: {pdf_path}")
        return pdf_path

    except Exception as e:
        print(f"❌ PDF generation error: {e}")
        return None


def play_alert_sound():
    """
    Play alert sound (beep) - HuggingFace compatible
    """
    if not SOUND_AVAILABLE:
        print("🔇 Sound disabled (cloud deployment)")
        return

    try:
        frequency = 1000  # Hz
        duration = 500  # milliseconds

        sample_rate = 22050
        samples = int(sample_rate * duration / 1000)
        wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration / 1000, samples))
        wave = (wave * 32767).astype(np.int16)

        stereo_wave = np.column_stack((wave, wave))

        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
        time.sleep(duration / 1000)
    except Exception as e:
        print(f"🔇 Sound error (expected on cloud): {e}")


def log_alert(alert_message, object_counts, image=None):
    """
    Log alert to file and global list
    """
    global alert_log

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert_entry = {
        'timestamp': timestamp,
        'message': alert_message,
        'object_counts': object_counts
    }

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
    """
    Check if any alert rules are triggered
    """
    triggered_alerts = []

    if alert_rules['person_count_enabled'] and 'person' in object_counts:
        if object_counts['person'] >= alert_rules['person_threshold']:
            triggered_alerts.append(
                f"⚠️ {object_counts['person']} Persons detected! (Threshold: {alert_rules['person_threshold']})")

    if alert_rules['car_alert_enabled'] and 'car' in object_counts:
        triggered_alerts.append(f"🚗 Car detected! (Count: {object_counts['car']})")

    if alert_rules['truck_alert_enabled'] and 'truck' in object_counts:
        triggered_alerts.append(f"🚚 Truck detected! (Count: {object_counts['truck']})")

    if alert_rules['dog_alert_enabled'] and 'dog' in object_counts:
        triggered_alerts.append(f"🐕 Dog detected! (Count: {object_counts['dog']})")

    if alert_rules['cat_alert_enabled'] and 'cat' in object_counts:
        triggered_alerts.append(f"🐈 Cat detected! (Count: {object_counts['cat']})")

    if alert_rules['bicycle_alert_enabled'] and 'bicycle' in object_counts:
        triggered_alerts.append(f"🚲 Bicycle detected! (Count: {object_counts['bicycle']})")

    total_objects = sum(object_counts.values())
    if alert_rules['total_objects_enabled'] and total_objects >= alert_rules['total_objects_threshold']:
        triggered_alerts.append(
            f"📊 {total_objects} Total objects detected! (Threshold: {alert_rules['total_objects_threshold']})")

    return triggered_alerts


def get_alert_log_display():
    """
    Get formatted alert log for display
    """
    if not alert_log:
        return "### 📂 No Alerts Yet\n\nAlerts will appear here when triggered!"

    display = "# 🚨 Alert Log (Last 20)\n\n"

    for idx, alert in enumerate(alert_log[:20]):
        display += f"## {idx + 1}. {alert['timestamp']}\n"
        display += f"**{alert['message']}**\n\n"

        if alert['object_counts']:
            display += "**Objects detected:**\n"
            for obj, count in alert['object_counts'].items():
                display += f"- {obj.capitalize()}: {count}\n"

        display += "\n---\n\n"

    return display


def clear_alert_log():
    """
    Clear all alerts
    """
    global alert_log
    alert_log = []

    for file in ALERTS_DIR.glob("*"):
        file.unlink()

    return "### ✅ Alert Log Cleared!\n\nAll alerts have been deleted."


def export_alert_log():
    """
    Export alert log as JSON
    """
    if not alert_log:
        return None

    export_data = {
        'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_alerts': len(alert_log),
        'alerts': alert_log
    }

    json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(export_data, json_temp, indent=2)
    json_temp.close()

    return json_temp.name


def save_to_history(image, object_counts, timestamp, model_name):
    """
    Save detection to history with model info
    """
    try:
        img_filename = f"detection_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        img_path = HISTORY_DIR / img_filename
        cv2.imwrite(str(img_path), image)

        metadata = {
            'timestamp': timestamp,
            'model': model_name,
            'object_counts': object_counts,
            'total_objects': sum(object_counts.values()),
            'image_path': str(img_path)
        }

        meta_filename = f"metadata_{timestamp.replace(':', '-').replace(' ', '_')}.json"
        meta_path = HISTORY_DIR / meta_filename
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False


def load_history():
    """
    Load detection history
    """
    try:
        history_files = sorted(HISTORY_DIR.glob("metadata_*.json"), reverse=True)

        if not history_files:
            return "### 📂 No History Yet\n\nStart detecting objects to build your history!"

        history_text = "# 📂 Detection History\n\n"

        for idx, meta_file in enumerate(history_files[:10]):
            with open(meta_file, 'r') as f:
                data = json.load(f)

            history_text += f"## 🕒 {data['timestamp']}\n"
            history_text += f"**Model:** {data.get('model', 'yolo11n.pt')}\n"
            history_text += f"**Total Objects:** {data['total_objects']}\n\n"

            for obj, count in sorted(data['object_counts'].items(), key=lambda x: x[1], reverse=True):
                history_text += f"- {obj.capitalize()}: {count}\n"

            history_text += "\n---\n\n"

        return history_text
    except Exception as e:
        return f"### ⚠️ Error Loading History\n\n{str(e)}"


def clear_history():
    """
    Clear all history
    """
    try:
        for file in HISTORY_DIR.glob("*"):
            file.unlink()
        return "### ✅ History Cleared!\n\nAll detection history has been deleted."
    except Exception as e:
        return f"### ⚠️ Error: {str(e)}"


def detect_objects_with_alerts(image, confidence, use_custom_colors,
                               person_alert_enabled, person_threshold,
                               car_alert, truck_alert, dog_alert, cat_alert, bicycle_alert,
                               total_objects_enabled, total_threshold,
                               sound_enabled, log_enabled,
                               model_name,
                               *filters):
    """
    Detect objects with smart alerts and model selection
    """
    if image is None:
        return None, "⚠️ Please upload an image first!", None, None, None, None, None, None

    # Load selected model
    load_model(model_name)

    filter_classes = [
        'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
        'dog', 'cat', 'bird', 'bottle', 'cup', 'laptop', 'phone', 'chair'
    ]

    selected_filters = [filter_classes[i] for i, f in enumerate(filters) if f]

    start_time = time.time()
    results = current_model.predict(source=image, conf=confidence, save=False)
    process_time = time.time() - start_time

    if use_custom_colors:
        annotated_image = draw_custom_boxes(image, results, confidence)
    else:
        annotated_image = results[0].plot()

    # Generate heatmap
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

        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

        export_data.append({
            'object': class_name,
            'confidence': f"{conf:.4f}",
            'bbox_x1': round(bbox[0], 2),
            'bbox_y1': round(bbox[1], 2),
            'bbox_x2': round(bbox[2], 2),
            'bbox_y2': round(bbox[3], 2)
        })

    # Check alerts
    alert_rules = {
        'person_count_enabled': person_alert_enabled,
        'person_threshold': person_threshold,
        'car_alert_enabled': car_alert,
        'truck_alert_enabled': truck_alert,
        'dog_alert_enabled': dog_alert,
        'cat_alert_enabled': cat_alert,
        'bicycle_alert_enabled': bicycle_alert,
        'total_objects_enabled': total_objects_enabled,
        'total_objects_threshold': total_threshold
    }

    triggered_alerts = check_alerts(object_counts, alert_rules)

    # Handle alerts
    alert_summary = ""
    if triggered_alerts:
        alert_summary = "\n\n## 🚨 ALERTS TRIGGERED!\n\n"

        for alert_msg in triggered_alerts:
            alert_summary += f"### {alert_msg}\n\n"

            if sound_enabled:
                threading.Thread(target=play_alert_sound).start()

            if log_enabled:
                log_alert(alert_msg, object_counts, annotated_image)

        alert_summary += "---\n\n"

    total = len(detections)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to history
    if object_counts:
        save_to_history(annotated_image, object_counts, timestamp, model_name)

    summary = f"# 📊 Detection Results\n\n"
    summary += f"**🤖 Model:** {model_name}\n"

    if use_custom_colors:
        summary += "**🎨 Custom Colors:** ON\n"

    if selected_filters:
        summary += f"**🎯 Filtered to:** {', '.join(selected_filters)}\n"
    else:
        summary += f"**🎯 Filter:** All objects\n"

    summary += f"**Timestamp:** {timestamp}\n"
    summary += f"**⏱️ Processing Time:** {process_time:.2f} seconds\n"
    summary += f"**🚀 Speed:** {1 / process_time:.1f} FPS\n"

    summary += alert_summary

    summary += f"## 🎯 Total Objects Found: **{total}**\n\n"

    if total > 0:
        summary += "### 📈 Live Object Counter:\n\n"

        emoji_map = {
            'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌',
            'bicycle': '🚲', 'motorcycle': '🏍️', 'dog': '🐕', 'cat': '🐈',
            'bird': '🐦', 'horse': '🐴', 'bottle': '🍾', 'cup': '☕',
            'laptop': '💻', 'phone': '📱', 'book': '📚', 'chair': '🪑',
            'traffic light': '🚦', 'stop sign': '🛑',
        }

        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)

        for obj, count in sorted_objects:
            emoji = emoji_map.get(obj, '📦')
            bar = '🟦' * count
            summary += f"**{emoji} {obj.capitalize()}:** {count} {bar}\n\n"

        summary += "\n---\n\n### 📋 Detailed Detections:\n\n"
        for i, det in enumerate(detections, 1):
            summary += f"{i}. {det}\n"
    else:
        summary = f"# ⚠️ No objects detected!\n\n**Model:** {model_name}\n\n"
        if selected_filters:
            summary += f"**Filter active:** Looking for {', '.join(selected_filters)}\n\n"
        summary += "**Try:** Lowering confidence, changing filter, or using a different image"

    json_file = None
    csv_file = None
    pdf_file = None

    if export_data:
        # JSON Export
        json_output = {
            'timestamp': timestamp,
            'model': model_name,
            'custom_colors': use_custom_colors,
            'applied_filters': selected_filters if selected_filters else 'all',
            'processing_time_seconds': round(process_time, 2),
            'fps': round(1 / process_time, 2),
            'total_objects': total,
            'object_summary': object_counts,
            'alerts_triggered': triggered_alerts,
            'detections': export_data
        }

        json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(json_output, json_temp, indent=2)
        json_temp.close()
        json_file = json_temp.name

        # CSV Export
        csv_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        csv_temp.write("Object,Confidence,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2\n")
        for item in export_data:
            csv_temp.write(
                f"{item['object']},{item['confidence']},{item['bbox_x1']},{item['bbox_y1']},{item['bbox_x2']},{item['bbox_y2']}\n")
        csv_temp.close()
        csv_file = csv_temp.name

        # PDF Export
        pdf_metadata = {
            'timestamp': timestamp,
            'model': model_name,
            'confidence': confidence,
            'total_objects': total,
            'processing_time': process_time,
            'fps': 1 / process_time
        }

        pdf_file = generate_pdf_report(annotated_image, export_data, object_counts, pdf_metadata)

    alert_log_display = get_alert_log_display()

    return annotated_image, summary, alert_log_display, json_file, csv_file, pdf_file, heatmap_pure, heatmap_overlay


def detect_batch_images(images, confidence, use_custom_colors, model_name, progress=gr.Progress()):
    """
    Batch process multiple images with model selection
    """
    if not images or len(images) == 0:
        return None, "⚠️ Please upload at least one image!", None

    # Load selected model
    load_model(model_name)

    progress(0, desc="Starting batch processing...")

    all_results = []
    all_images = []
    total_objects = 0
    combined_counts = {}

    for idx, img_file in enumerate(images):
        progress((idx + 1) / len(images), desc=f"Processing image {idx + 1}/{len(images)}")

        image = Image.open(img_file.name)
        results = current_model.predict(source=image, conf=confidence, save=False)

        if use_custom_colors:
            annotated = draw_custom_boxes(image, results, confidence)
        else:
            annotated = results[0].plot()

        all_images.append(annotated)

        image_counts = {}
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if class_name in image_counts:
                image_counts[class_name] += 1
            else:
                image_counts[class_name] = 1

            if class_name in combined_counts:
                combined_counts[class_name] += 1
            else:
                combined_counts[class_name] = 1

            total_objects += 1

        all_results.append({
            'image_number': idx + 1,
            'objects_found': len(results[0].boxes),
            'object_breakdown': image_counts
        })

    summary = f"# 📦 Batch Processing Complete!\n\n"
    summary += f"**🤖 Model:** {model_name}\n"
    summary += f"**Total Images Processed:** {len(images)}\n"
    summary += f"**Total Objects Found:** {total_objects}\n\n"
    summary += f"### 📊 Combined Object Counts:\n\n"

    for obj, count in sorted(combined_counts.items(), key=lambda x: x[1], reverse=True):
        summary += f"- **{obj.capitalize()}:** {count}\n"

    summary += f"\n### 📋 Per-Image Results:\n\n"
    for result in all_results:
        summary += f"**Image {result['image_number']}:** {result['objects_found']} objects\n"
        for obj, count in result['object_breakdown'].items():
            summary += f"  - {obj}: {count}\n"
        summary += "\n"

    zip_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    with zipfile.ZipFile(zip_temp.name, 'w') as zipf:
        for idx, img in enumerate(all_images):
            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(img_temp.name, img)
            zipf.write(img_temp.name, f'detected_image_{idx + 1}.jpg')
            os.unlink(img_temp.name)

    progress(1.0, desc="Done!")

    return all_images[0] if all_images else None, summary, zip_temp.name


def detect_objects_video_with_frames(video, confidence, model_name, progress=gr.Progress()):
    """
    Detect objects in video with frame extraction and model selection
    """
    if video is None:
        return None, "⚠️ Please upload a video first!", None, None, None

    # Load selected model
    load_model(model_name)

    start_time = time.time()
    progress(0, desc="Starting video processing...")

    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    all_detections = {}
    video_export_data = []
    frame_detection_counts = []
    key_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")

        results = current_model.predict(source=frame, conf=confidence, save=False, verbose=False)
        annotated_frame = results[0].plot()

        num_detections = len(results[0].boxes)
        frame_detection_counts.append((frame_count, num_detections))

        if num_detections > 0 and len(key_frames) < 5:
            key_frame_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(key_frame_temp.name, annotated_frame)
            key_frames.append((frame_count, num_detections, key_frame_temp.name))

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            conf = float(box.conf[0])

            if class_name in all_detections:
                all_detections[class_name] += 1
            else:
                all_detections[class_name] = 1

            video_export_data.append({
                'frame': frame_count,
                'object': class_name,
                'confidence': f"{conf:.4f}"
            })

        out.write(annotated_frame)

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_detections = sum(all_detections.values())

    max_detections_frame = max(frame_detection_counts, key=lambda x: x[1]) if frame_detection_counts else (0, 0)

    summary = f"# 🎥 Video Processing Complete!\n\n"
    summary += f"**🤖 Model:** {model_name}\n"
    summary += f"**Timestamp:** {timestamp}\n"
    summary += f"**⏱️ Total Processing Time:** {total_time:.2f} seconds\n"
    summary += f"**🚀 Average Speed:** {avg_fps:.1f} FPS\n"
    summary += f"**📹 Total Frames:** {frame_count}\n"
    summary += f"**🎯 Total Detections:** {total_detections}\n"
    summary += f"**📊 Peak Frame:** #{max_detections_frame[0]} ({max_detections_frame[1]} objects)\n\n"

    if all_detections:
        summary += "### 📊 Objects Detected:\n\n"
        emoji_map = {
            'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌',
            'dog': '🐕', 'cat': '🐈', 'bicycle': '🚲', 'motorcycle': '🏍️'
        }

        for obj, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True):
            emoji = emoji_map.get(obj, '📦')
            bar = '🟩' * min(count // 10, 20)
            summary += f"**{emoji} {obj.capitalize()}:** {count} times {bar}\n\n"

        summary += f"\n### 🎬 Key Frames Extracted: {len(key_frames)}\n"
        for frame_num, num_det, _ in key_frames:
            summary += f"- Frame #{frame_num}: {num_det} objects\n"
    else:
        summary = f"### ⚠️ No objects detected!\n**Model:** {model_name}"

    progress(1.0, desc="Done!")

    json_file = None
    csv_file = None

    if video_export_data:
        json_output = {
            'timestamp': timestamp,
            'model': model_name,
            'processing_time_seconds': round(total_time, 2),
            'average_fps': round(avg_fps, 2),
            'total_frames': frame_count,
            'total_detections': total_detections,
            'peak_frame': max_detections_frame[0],
            'peak_detections': max_detections_frame[1],
            'object_summary': all_detections,
            'frame_detections': video_export_data
        }

        json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(json_output, json_temp, indent=2)
        json_temp.close()
        json_file = json_temp.name

        csv_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        csv_temp.write("Frame,Object,Confidence\n")
        for item in video_export_data:
            csv_temp.write(f"{item['frame']},{item['object']},{item['confidence']}\n")
        csv_temp.close()
        csv_file = csv_temp.name

    preview_frame = key_frames[0][2] if key_frames else None

    return output_path, summary, preview_frame, json_file, csv_file


frame_times = []


def detect_webcam(image, confidence, model_name):
    """
    Real-time webcam detection with FPS counter and model selection
    """
    global frame_times

    if image is None:
        return None, "### 🎯 Waiting for webcam..."

    # Load selected model
    load_model(model_name)

    current_time = time.time()
    frame_times.append(current_time)

    frame_times = [t for t in frame_times if current_time - t < 1.0]
    fps = len(frame_times)

    results = current_model.predict(source=image, conf=confidence, save=False, verbose=False)
    annotated_image = results[0].plot()

    object_counts = {}
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]

        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

    counter_text = "### 🔴 LIVE Detection\n\n"
    counter_text += f"**🤖 Model:** {model_name}\n"
    counter_text += f"**🚀 FPS: {fps}** | **⏱️ Latency: {1000 / fps if fps > 0 else 0:.0f}ms**\n\n"

    if object_counts:
        emoji_map = {
            'person': '👤', 'car': '🚗', 'truck': '🚚', 'bus': '🚌',
            'dog': '🐕', 'cat': '🐈', 'bicycle': '🚲', 'motorcycle': '🏍️',
            'bottle': '🍾', 'cup': '☕', 'laptop': '💻', 'phone': '📱'
        }

        total = sum(object_counts.values())
        counter_text += f"**🎯 Total: {total} objects**\n\n"

        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            emoji = emoji_map.get(obj, '📦')
            counter_text += f"{emoji} **{obj.capitalize()}: {count}** | "

        counter_text = counter_text.rstrip(" | ")
    else:
        counter_text += "*No objects detected!*"

    return annotated_image, counter_text


# Create Gradio interface
with gr.Blocks(title="YOLO Object Detection Pro - Multi-Model") as demo:
    gr.Markdown("# 🚀 YOLO Object Detection Pro - Multi-Model")
    gr.Markdown(
        "### AI-powered detection with **Multi-Model Comparison**, custom colors, batch processing, video analysis, history, **SMART ALERTS**, **PDF Reports** & **🗺️ HEATMAPS**!")

    # GLOBAL MODEL SELECTOR
    with gr.Row():
        with gr.Column(scale=3):
            global_model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value="yolo11n.pt",
                label="🤖 Select YOLO Model (Global)",
                info="n=fastest ⚡ | s=fast 🏃 | m=balanced ⚖️ | l=accurate 🎯 | x=most accurate 🔥"
            )
        with gr.Column(scale=2):
            gr.Markdown("""
            ### Model Guide:
            - **n**: 2.6M params, fastest
            - **s**: 9.4M params, very fast
            - **m**: 20.1M params, balanced
            - **l**: 25.3M params, accurate
            - **x**: 56.9M params, best accuracy
            """)

    with gr.Tabs():
        # Image Detection Tab WITH ALERTS
        with gr.Tab("📷 Image Detection + Alerts"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="📤 Upload Image")
                    image_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence Threshold")

                    custom_colors_toggle = gr.Checkbox(label="🎨 Use Custom Colors", value=True,
                                                       info="Different color per object type")

                    # ALERT SETTINGS
                    with gr.Accordion("🔔 Smart Alert Settings", open=True):
                        gr.Markdown("**Configure detection alerts:**")

                        with gr.Row():
                            person_alert_enabled = gr.Checkbox(label="Alert on Person Count", value=False)
                            person_threshold = gr.Slider(1, 20, 3, step=1, label="Person Threshold")

                        with gr.Row():
                            total_objects_enabled = gr.Checkbox(label="Alert on Total Objects", value=False)
                            total_threshold = gr.Slider(1, 50, 10, step=1, label="Total Objects Threshold")

                        gr.Markdown("**Specific Object Alerts:**")
                        with gr.Row():
                            car_alert = gr.Checkbox(label="🚗 Alert on Car", value=False)
                            truck_alert = gr.Checkbox(label="🚚 Alert on Truck", value=False)
                            bicycle_alert = gr.Checkbox(label="🚲 Alert on Bicycle", value=False)

                        with gr.Row():
                            dog_alert = gr.Checkbox(label="🐕 Alert on Dog", value=False)
                            cat_alert = gr.Checkbox(label="🐈 Alert on Cat", value=False)

                        gr.Markdown("**Notification Methods:**")
                        with gr.Row():
                            sound_enabled = gr.Checkbox(label="🔊 Sound Alert", value=True,
                                                        info="Auto-disabled on cloud")
                            log_enabled = gr.Checkbox(label="📝 Log Alert", value=True)

                    # Object filter checkboxes
                    with gr.Accordion("🎯 Filter Objects (Optional)", open=False):
                        gr.Markdown("**Select specific objects:**")

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

                    image_btn = gr.Button("🔍 Detect with Alerts", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["https://ultralytics.com/images/bus.jpg"],
                            ["https://ultralytics.com/images/zidane.jpg"],
                        ],
                        inputs=image_input,
                        label="📸 Examples"
                    )

                with gr.Column():
                    image_output = gr.Image(type="numpy", label="✨ Detection Results")
                    image_text = gr.Markdown()

                    # Heatmap visualizations
                    with gr.Accordion("🗺️ Heatmap Visualization", open=False):
                        with gr.Row():
                            heatmap_pure = gr.Image(type="numpy", label="🔥 Pure Heatmap")
                            heatmap_overlay = gr.Image(type="numpy", label="🎨 Heatmap Overlay")
                        gr.Markdown("""
                        **Heatmap Guide:**
                        - 🔵 Blue = Low detection density
                        - 🟢 Green = Medium density
                        - 🟡 Yellow = High density
                        - 🔴 Red = Highest density (hotspots)
                        """)

                    alert_log_display = gr.Markdown("### 📂 Alert Log\n\nAlerts will appear here!")

                    with gr.Row():
                        json_download = gr.File(label="📄 JSON")
                        csv_download = gr.File(label="📊 CSV")
                        pdf_download = gr.File(label="📋 PDF Report")

            image_btn.click(
                fn=detect_objects_with_alerts,
                inputs=[
                    image_input, image_confidence, custom_colors_toggle,
                    person_alert_enabled, person_threshold,
                    car_alert, truck_alert, dog_alert, cat_alert, bicycle_alert,
                    total_objects_enabled, total_threshold,
                    sound_enabled, log_enabled,
                    global_model_selector,
                    filter_person, filter_car, filter_truck, filter_bus,
                    filter_bicycle, filter_motorcycle, filter_dog, filter_cat,
                    filter_bird, filter_bottle, filter_cup, filter_laptop,
                    filter_phone, filter_chair
                ],
                outputs=[image_output, image_text, alert_log_display, json_download, csv_download, pdf_download,
                         heatmap_pure, heatmap_overlay]
            )

        # Batch Processing Tab
        with gr.Tab("📦 Batch Processing"):
            with gr.Row():
                with gr.Column():
                    batch_images = gr.File(file_count="multiple", label="📤 Upload Multiple Images",
                                           file_types=["image"])
                    batch_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    batch_colors = gr.Checkbox(label="🎨 Use Custom Colors", value=True)
                    batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")

                    gr.Markdown("""
                    **Batch Processing:**
                    - Upload multiple images at once
                    - Compare models on same images
                    - Download zip with all results
                    """)

                with gr.Column():
                    batch_preview = gr.Image(type="numpy", label="Preview (First Image)")
                    batch_text = gr.Markdown()
                    batch_zip = gr.File(label="📦 Download All Results (ZIP)")

            batch_btn.click(
                fn=detect_batch_images,
                inputs=[batch_images, batch_confidence, batch_colors, global_model_selector],
                outputs=[batch_preview, batch_text, batch_zip]
            )

        # Video Detection Tab
        with gr.Tab("🎥 Video Detection"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="📤 Upload Video")
                    video_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    video_btn = gr.Button("🎬 Process Video", variant="primary", size="lg")
                    gr.Markdown("**Note:** Extracts key frames with most detections!")

                with gr.Column():
                    video_output = gr.Video(label="✨ Detected Video")
                    video_text = gr.Markdown()

                    key_frame_preview = gr.Image(type="filepath", label="🎬 Key Frame Preview")

                    with gr.Row():
                        video_json = gr.File(label="📄 JSON")
                        video_csv = gr.File(label="📊 CSV")

            video_btn.click(
                fn=detect_objects_video_with_frames,
                inputs=[video_input, video_confidence, global_model_selector],
                outputs=[video_output, video_text, key_frame_preview, video_json, video_csv]
            )

        # Video Heatmap Tab
        with gr.Tab("🗺️ Video Heatmap"):
            gr.Markdown("### 🔥 Generate Detection Heatmap from Video")
            gr.Markdown("**Shows hotspots where objects are detected most frequently across all frames!**")

            with gr.Row():
                with gr.Column():
                    heatmap_video_input = gr.Video(label="📤 Upload Video")
                    heatmap_video_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence Threshold")
                    heatmap_video_btn = gr.Button("🗺️ Generate Heatmap", variant="primary", size="lg")

                    gr.Markdown("""
                    **Use Cases:**
                    - 🏢 Security: Find high-traffic areas
                    - 🚗 Traffic: Identify busy lanes
                    - 🛒 Retail: Customer movement patterns
                    - ⚽ Sports: Player positioning analysis
                    """)

                with gr.Column():
                    with gr.Row():
                        heatmap_video_pure = gr.Image(type="numpy", label="🔥 Pure Heatmap")
                        heatmap_video_overlay = gr.Image(type="numpy", label="🎨 Overlay")

                    gr.Markdown("""
                    **Legend:**
                    - 🔵 **Blue**: Low activity
                    - 🟢 **Green**: Moderate activity
                    - 🟡 **Yellow**: High activity
                    - 🔴 **Red**: Hotspot (highest activity)
                    """)

            heatmap_video_btn.click(
                fn=generate_video_heatmap,
                inputs=[heatmap_video_input, heatmap_video_confidence, global_model_selector],
                outputs=[heatmap_video_pure, heatmap_video_overlay]
            )

        # Webcam Tab
        with gr.Tab("📸 Live Webcam"):
            gr.Markdown("### 🔴 Real-Time Detection")
            gr.Markdown("**Click 'Start Webcam' and allow camera access!**")

            with gr.Row():
                with gr.Column():
                    webcam_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence")
                    webcam_counter = gr.Markdown("### 🎯 Waiting...")

                    gr.Markdown("**Tips:** Good lighting, center objects, adjust confidence")

                with gr.Column():
                    webcam_output = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="🎥 Live Feed"
                    )

            webcam_output.stream(
                fn=detect_webcam,
                inputs=[webcam_output, webcam_confidence, global_model_selector],
                outputs=[webcam_output, webcam_counter],
                time_limit=60,
                stream_every=0.1
            )

        # Alert Log Tab
        with gr.Tab("🚨 Alert Log"):
            gr.Markdown("### 🚨 Alert History & Management")
            gr.Markdown("View and manage all triggered alerts!")

            with gr.Row():
                alert_refresh_btn = gr.Button("🔄 Refresh Log", variant="primary")
                alert_clear_btn = gr.Button("🗑️ Clear All Alerts", variant="stop")
                alert_export_btn = gr.Button("📥 Export Log", variant="secondary")

            alert_history_display = gr.Markdown("Click 'Refresh Log' to view alerts!")
            alert_export_file = gr.File(label="📄 Download Alert Log")

            alert_refresh_btn.click(fn=get_alert_log_display, outputs=alert_history_display)
            alert_clear_btn.click(fn=clear_alert_log, outputs=alert_history_display)
            alert_export_btn.click(fn=export_alert_log, outputs=alert_export_file)

        # History Tab
        with gr.Tab("📂 Detection History"):
            gr.Markdown("### 📂 Detection History")
            gr.Markdown("View all your past detections with timestamps, models, and object counts!")

            with gr.Row():
                history_load_btn = gr.Button("🔄 Load History", variant="primary")
                history_clear_btn = gr.Button("🗑️ Clear History", variant="stop")

            history_display = gr.Markdown("Click 'Load History' to view your past detections!")

            history_load_btn.click(fn=load_history, outputs=history_display)
            history_clear_btn.click(fn=clear_history, outputs=history_display)

    gr.Markdown("---")
    gr.Markdown("Built by **Samith Shivakumar** | Powered by YOLOv11 🚀")
    gr.Markdown(
        "⭐ **Features:** 🤖 **Multi-Model** | 🎨 Custom Colors | 📦 Batch | 🎬 Video Analysis | 📸 Webcam | 📂 History | 🔔 **Smart Alerts** | 📋 **PDF Reports** | 🗺️ **Heatmaps**")

if __name__ == "__main__":
    demo.launch()
