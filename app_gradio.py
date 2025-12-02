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

# Load YOLO model
model = YOLO("yolo11n.pt")


def detect_objects_image(image, confidence):
    """
    Detect objects in an image using YOLO with live counter, export, and timing
    """
    if image is None:
        return None, "⚠️ Please upload an image first!", None, None

    # Start timer
    start_time = time.time()

    # Run detection
    results = model.predict(source=image, conf=confidence, save=False)

    # Calculate processing time
    process_time = time.time() - start_time

    # Get annotated image
    annotated_image = results[0].plot()

    # Count objects and collect detailed data
    detections = []
    object_counts = {}
    export_data = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])
        bbox = box.xyxy[0].tolist()

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

    # Format results
    total = len(detections)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = f"# 📊 Detection Results\n\n"
    summary += f"**Timestamp:** {timestamp}\n"
    summary += f"**⏱️ Processing Time:** {process_time:.2f} seconds\n"
    summary += f"**🚀 Speed:** {1 / process_time:.1f} FPS\n\n"
    summary += f"## 🎯 Total Objects: **{total}**\n\n"

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
        summary = "# ⚠️ No objects detected!\n\n"
        summary += "**Try:** Lowering confidence or using a different image"

    # Create export files
    json_file = None
    csv_file = None

    if export_data:
        json_output = {
            'timestamp': timestamp,
            'processing_time_seconds': round(process_time, 2),
            'fps': round(1 / process_time, 2),
            'total_objects': total,
            'object_summary': object_counts,
            'detections': export_data
        }

        json_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(json_output, json_temp, indent=2)
        json_temp.close()
        json_file = json_temp.name

        csv_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        csv_temp.write("Object,Confidence,BBox_X1,BBox_Y1,BBox_X2,BBox_Y2\n")
        for item in export_data:
            csv_temp.write(
                f"{item['object']},{item['confidence']},{item['bbox_x1']},{item['bbox_y1']},{item['bbox_x2']},{item['bbox_y2']}\n")
        csv_temp.close()
        csv_file = csv_temp.name

    return annotated_image, summary, json_file, csv_file


def detect_objects_video(video, confidence, progress=gr.Progress()):
    """
    Detect objects in a video with timing stats
    """
    if video is None:
        return None, "⚠️ Please upload a video first!", None, None

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")

        results = model.predict(source=frame, conf=confidence, save=False, verbose=False)
        annotated_frame = results[0].plot()

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

    summary = f"# 🎥 Video Processing Complete!\n\n"
    summary += f"**Timestamp:** {timestamp}\n"
    summary += f"**⏱️ Total Processing Time:** {total_time:.2f} seconds\n"
    summary += f"**🚀 Average Speed:** {avg_fps:.1f} FPS\n"
    summary += f"**📹 Total Frames:** {frame_count}\n"
    summary += f"**🎯 Total Detections:** {total_detections}\n\n"

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
    else:
        summary = "### ⚠️ No objects detected!"

    progress(1.0, desc="Done!")

    # Create exports
    json_file = None
    csv_file = None

    if video_export_data:
        json_output = {
            'timestamp': timestamp,
            'processing_time_seconds': round(total_time, 2),
            'average_fps': round(avg_fps, 2),
            'total_frames': frame_count,
            'total_detections': total_detections,
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

    return output_path, summary, json_file, csv_file


# Global variables for FPS tracking
frame_times = []


def detect_webcam(image, confidence):
    """
    Real-time webcam detection with FPS counter
    """
    global frame_times

    if image is None:
        return None, "### 🎯 Waiting for webcam..."

    # Track FPS
    current_time = time.time()
    frame_times.append(current_time)

    # Keep only last 30 frames for FPS calculation
    frame_times = [t for t in frame_times if current_time - t < 1.0]
    fps = len(frame_times)

    # Run detection
    results = model.predict(source=image, conf=confidence, save=False, verbose=False)
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
with gr.Blocks(title="YOLO Object Detection") as demo:
    gr.Markdown("# 🎯 YOLO Object Detection")
    gr.Markdown("### AI-powered detection with real-time stats, data export, and performance metrics!")

    with gr.Tabs():
        # Image Detection Tab
        with gr.Tab("📷 Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="📤 Upload Image")
                    image_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence Threshold")
                    image_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["https://ultralytics.com/images/bus.jpg"],
                            ["https://ultralytics.com/images/zidane.jpg"],
                        ],
                        inputs=image_input,
                        label="📸 Try Examples"
                    )

                with gr.Column():
                    image_output = gr.Image(type="numpy", label="✨ Detection Results")
                    image_text = gr.Markdown()

                    with gr.Row():
                        json_download = gr.File(label="📄 Download JSON")
                        csv_download = gr.File(label="📊 Download CSV")

            image_btn.click(
                fn=detect_objects_image,
                inputs=[image_input, image_confidence],
                outputs=[image_output, image_text, json_download, csv_download]
            )

        # Video Detection Tab
        with gr.Tab("🎥 Video Detection"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="📤 Upload Video")
                    video_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence Threshold")
                    video_btn = gr.Button("🎬 Process Video", variant="primary", size="lg")
                    gr.Markdown("**Note:** Video processing may take 1-2 minutes!")

                with gr.Column():
                    video_output = gr.Video(label="✨ Detection Results")
                    video_text = gr.Markdown()

                    with gr.Row():
                        video_json_download = gr.File(label="📄 Download JSON")
                        video_csv_download = gr.File(label="📊 Download CSV")

            video_btn.click(
                fn=detect_objects_video,
                inputs=[video_input, video_confidence],
                outputs=[video_output, video_text, video_json_download, video_csv_download]
            )

        # Webcam Tab with FPS
        with gr.Tab("📸 Live Webcam"):
            gr.Markdown("### 🔴 Real-Time Detection with FPS Counter")
            gr.Markdown("**Click 'Start Webcam' and allow camera access!**")

            with gr.Row():
                with gr.Column():
                    webcam_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence Threshold")
                    webcam_counter = gr.Markdown("### 🎯 Waiting for webcam...")

                    gr.Markdown("""
                    **Performance Tips:**
                    - Good lighting improves FPS
                    - Close other apps for better performance
                    - Adjust confidence to balance speed/accuracy
                    """)

                with gr.Column():
                    webcam_output = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="🎥 Live Feed"
                    )

            webcam_output.stream(
                fn=detect_webcam,
                inputs=[webcam_output, webcam_confidence],
                outputs=[webcam_output, webcam_counter],
                time_limit=60,
                stream_every=0.1
            )

    gr.Markdown("---")
    gr.Markdown("Built by **Samith Shivakumar** | Powered by YOLOv11 🚀")
    gr.Markdown("⭐ **Features:** Webcam + Object Counter + Data Export + **Performance Stats (FPS/Timing)**")

if __name__ == "__main__":
    demo.launch()
