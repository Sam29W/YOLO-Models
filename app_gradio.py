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

# Load YOLO model
model = YOLO("yolo11n.pt")

# Create history directory
HISTORY_DIR = Path("detection_history")
HISTORY_DIR.mkdir(exist_ok=True)


def draw_custom_boxes(image, results, confidence):
    """
    Draw custom colored bounding boxes on image
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Make a copy
    annotated = image.copy()

    # Custom color map (BGR format for OpenCV)
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

    # Draw boxes
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])

        if conf < confidence:
            continue

        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Get color
        color = color_map.get(class_name, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Draw label background
        label = f"{class_name}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def save_to_history(image, object_counts, timestamp):
    """
    Save detection to history
    """
    try:
        # Save image
        img_filename = f"detection_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        img_path = HISTORY_DIR / img_filename
        cv2.imwrite(str(img_path), image)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
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

        for idx, meta_file in enumerate(history_files[:10]):  # Last 10 detections
            with open(meta_file, 'r') as f:
                data = json.load(f)

            history_text += f"## 🕒 {data['timestamp']}\n"
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


def detect_objects_image_filtered(image, confidence, use_custom_colors, *filters):
    """
    Detect objects with custom colors
    """
    if image is None:
        return None, "⚠️ Please upload an image first!", None, None

    filter_classes = [
        'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
        'dog', 'cat', 'bird', 'bottle', 'cup', 'laptop', 'phone', 'chair'
    ]

    selected_filters = [filter_classes[i] for i, f in enumerate(filters) if f]

    start_time = time.time()
    results = model.predict(source=image, conf=confidence, save=False)
    process_time = time.time() - start_time

    if use_custom_colors:
        annotated_image = draw_custom_boxes(image, results, confidence)
    else:
        annotated_image = results[0].plot()

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

    total = len(detections)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to history
    if object_counts:
        save_to_history(annotated_image, object_counts, timestamp)

    summary = f"# 📊 Detection Results\n\n"

    if use_custom_colors:
        summary += "**🎨 Custom Colors:** ON\n"

    if selected_filters:
        summary += f"**🎯 Filtered to:** {', '.join(selected_filters)}\n"
    else:
        summary += f"**🎯 Filter:** All objects\n"

    summary += f"**Timestamp:** {timestamp}\n"
    summary += f"**⏱️ Processing Time:** {process_time:.2f} seconds\n"
    summary += f"**🚀 Speed:** {1 / process_time:.1f} FPS\n\n"
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
        summary = "# ⚠️ No objects detected!\n\n"
        if selected_filters:
            summary += f"**Filter active:** Looking for {', '.join(selected_filters)}\n\n"
        summary += "**Try:** Lowering confidence, changing filter, or using a different image"

    json_file = None
    csv_file = None

    if export_data:
        json_output = {
            'timestamp': timestamp,
            'custom_colors': use_custom_colors,
            'applied_filters': selected_filters if selected_filters else 'all',
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


def detect_batch_images(images, confidence, use_custom_colors, progress=gr.Progress()):
    """
    Batch process multiple images
    """
    if not images or len(images) == 0:
        return None, "⚠️ Please upload at least one image!", None

    progress(0, desc="Starting batch processing...")

    all_results = []
    all_images = []
    total_objects = 0
    combined_counts = {}

    for idx, img_file in enumerate(images):
        progress((idx + 1) / len(images), desc=f"Processing image {idx + 1}/{len(images)}")

        # Load image
        image = Image.open(img_file.name)

        results = model.predict(source=image, conf=confidence, save=False)

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


def detect_objects_video_with_frames(video, confidence, progress=gr.Progress()):
    """
    Detect objects in video with frame extraction
    """
    if video is None:
        return None, "⚠️ Please upload a video first!", None, None, None

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

        results = model.predict(source=frame, conf=confidence, save=False, verbose=False)
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
        summary = "### ⚠️ No objects detected!"

    progress(1.0, desc="Done!")

    json_file = None
    csv_file = None

    if video_export_data:
        json_output = {
            'timestamp': timestamp,
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


def detect_webcam(image, confidence):
    """
    Real-time webcam detection with FPS counter
    """
    global frame_times

    if image is None:
        return None, "### 🎯 Waiting for webcam..."

    current_time = time.time()
    frame_times.append(current_time)

    frame_times = [t for t in frame_times if current_time - t < 1.0]
    fps = len(frame_times)

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
with gr.Blocks(title="YOLO Object Detection Pro") as demo:
    gr.Markdown("# 🎯 YOLO Object Detection Pro")
    gr.Markdown("### AI-powered detection with custom colors, batch processing, video analysis, and history!")

    with gr.Tabs():
        # Image Detection Tab
        with gr.Tab("📷 Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="📤 Upload Image")
                    image_confidence = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Confidence Threshold")

                    custom_colors_toggle = gr.Checkbox(label="🎨 Use Custom Colors", value=True,
                                                       info="Different color per object type")

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

                    image_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")

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

                    with gr.Row():
                        json_download = gr.File(label="📄 JSON")
                        csv_download = gr.File(label="📊 CSV")

            image_btn.click(
                fn=detect_objects_image_filtered,
                inputs=[
                    image_input, image_confidence, custom_colors_toggle,
                    filter_person, filter_car, filter_truck, filter_bus,
                    filter_bicycle, filter_motorcycle, filter_dog, filter_cat,
                    filter_bird, filter_bottle, filter_cup, filter_laptop,
                    filter_phone, filter_chair
                ],
                outputs=[image_output, image_text, json_download, csv_download]
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
                    - All images processed automatically
                    - Download zip with all results
                    """)

                with gr.Column():
                    batch_preview = gr.Image(type="numpy", label="Preview (First Image)")
                    batch_text = gr.Markdown()
                    batch_zip = gr.File(label="📦 Download All Results (ZIP)")

            batch_btn.click(
                fn=detect_batch_images,
                inputs=[batch_images, batch_confidence, batch_colors],
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
                inputs=[video_input, video_confidence],
                outputs=[video_output, video_text, key_frame_preview, video_json, video_csv]
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
                inputs=[webcam_output, webcam_confidence],
                outputs=[webcam_output, webcam_counter],
                time_limit=60,
                stream_every=0.1
            )

        # History Tab
        with gr.Tab("📂 History"):
            gr.Markdown("### 📂 Detection History")
            gr.Markdown("View all your past detections with timestamps and object counts!")

            with gr.Row():
                history_load_btn = gr.Button("🔄 Load History", variant="primary")
                history_clear_btn = gr.Button("🗑️ Clear History", variant="stop")

            history_display = gr.Markdown("Click 'Load History' to view your past detections!")

            history_load_btn.click(fn=load_history, outputs=history_display)
            history_clear_btn.click(fn=clear_history, outputs=history_display)

    gr.Markdown("---")
    gr.Markdown("Built by **Samith Shivakumar** | Powered by YOLOv11 🚀")
    gr.Markdown("⭐ **Features:** 🎨 Custom Colors | 📦 Batch | 🎬 Video Analysis | 📸 Webcam | 📂 History")

if __name__ == "__main__":
    demo.launch()
