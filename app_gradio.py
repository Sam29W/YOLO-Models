import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# Load YOLO model
model = YOLO("yolo11n.pt")


def detect_objects_image(image, confidence):
    """
    Detect objects in an image using YOLO
    """
    if image is None:
        return None, "⚠️ Please upload an image first!"

    # Run detection
    results = model.predict(source=image, conf=confidence, save=False)

    # Get annotated image
    annotated_image = results[0].plot()

    # Count objects
    detections = []
    object_counts = {}

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        conf = float(box.conf[0])

        detections.append(f"{class_name}: {conf:.2%}")

        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

    # Format results
    total = len(detections)
    summary = f"### 📊 Detection Summary\n\n**Total Objects Found: {total}**\n\n"

    if total > 0:
        summary += "#### Object Breakdown:\n"
        for obj, count in object_counts.items():
            summary += f"- **{obj.capitalize()}**: {count}\n"
    else:
        summary = "### ⚠️ No objects detected!\n\nTry lowering the confidence threshold."

    return annotated_image, summary


def detect_objects_video(video, confidence, progress=gr.Progress()):
    """
    Detect objects in a video using YOLO
    """
    if video is None:
        return None, "⚠️ Please upload a video first!"

    progress(0, desc="Starting video processing...")

    # Open video
    cap = cv2.VideoCapture(video)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frame by frame
    frame_count = 0
    all_detections = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")

        # Run detection on frame
        results = model.predict(source=frame, conf=confidence, save=False, verbose=False)

        # Get annotated frame
        annotated_frame = results[0].plot()

        # Count objects in this frame
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if class_name in all_detections:
                all_detections[class_name] += 1
            else:
                all_detections[class_name] = 1

        # Write frame
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Create summary
    total_detections = sum(all_detections.values())
    summary = f"### 🎥 Video Processing Complete!\n\n"
    summary += f"**Total Frames Processed:** {frame_count}\n"
    summary += f"**Total Detections:** {total_detections}\n\n"

    if all_detections:
        summary += "#### Objects Detected Across Video:\n"
        for obj, count in sorted(all_detections.items(), key=lambda x: x[1], reverse=True):
            summary += f"- **{obj.capitalize()}**: {count} times\n"
    else:
        summary = "### ⚠️ No objects detected in video!\n\nTry lowering the confidence threshold."

    progress(1.0, desc="Done!")

    return output_path, summary


def detect_webcam(image, confidence):
    """
    Real-time webcam detection
    """
    if image is None:
        return None

    # Run detection
    results = model.predict(source=image, conf=confidence, save=False, verbose=False)

    # Get annotated image
    annotated_image = results[0].plot()

    return annotated_image


# Create Gradio interface with tabs
with gr.Blocks(title="YOLO Object Detection") as demo:
    gr.Markdown("# 🎯 YOLO Object Detection")
    gr.Markdown("### Detect objects in images, videos, or live from your webcam using YOLOv11")

    with gr.Tabs():
        # Image Detection Tab
        with gr.Tab("📷 Image Detection"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="📤 Upload Image")

                    image_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )

                    image_btn = gr.Button("🔍 Detect Objects", variant="primary", size="lg")

                    gr.Examples(
                        examples=[
                            ["https://ultralytics.com/images/bus.jpg"],
                            ["https://ultralytics.com/images/zidane.jpg"],
                        ],
                        inputs=image_input,
                        label="📸 Try Example Images"
                    )

                with gr.Column():
                    image_output = gr.Image(type="numpy", label="✨ Detection Results")
                    image_text = gr.Markdown()

            image_btn.click(
                fn=detect_objects_image,
                inputs=[image_input, image_confidence],
                outputs=[image_output, image_text]
            )

        # Video Detection Tab
        with gr.Tab("🎥 Video Detection"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="📤 Upload Video")

                    video_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )

                    video_btn = gr.Button("🎬 Process Video", variant="primary", size="lg")

                    gr.Markdown("**Note:** Video processing may take 1-2 minutes depending on length. Please wait!")

                with gr.Column():
                    video_output = gr.Video(label="✨ Detected Video")
                    video_text = gr.Markdown()

            video_btn.click(
                fn=detect_objects_video,
                inputs=[video_input, video_confidence],
                outputs=[video_output, video_text]
            )

        # NEW: Webcam Detection Tab
        with gr.Tab("📸 Live Webcam"):
            gr.Markdown("### 🔴 Real-Time Object Detection")
            gr.Markdown("**Click 'Start Webcam' below and allow camera access!**")

            with gr.Row():
                with gr.Column():
                    webcam_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Adjust detection sensitivity"
                    )

                    gr.Markdown("""
                    **Tips for best results:**
                    - Ensure good lighting
                    - Keep objects in center of frame
                    - Adjust confidence for more/fewer detections
                    - Works best with clear, unobstructed views
                    """)

            webcam_output = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="🎥 Live Detection Feed"
            )

            # Real-time detection on webcam feed
            webcam_output.stream(
                fn=detect_webcam,
                inputs=[webcam_output, webcam_confidence],
                outputs=[webcam_output],
                time_limit=60,
                stream_every=0.1
            )

    # Footer
    gr.Markdown("---")
    gr.Markdown("Built by **Samith Shivakumar** | Powered by YOLOv11 🚀")
    gr.Markdown("⭐ **NEW:** Real-time webcam detection added!")

if __name__ == "__main__":
    demo.launch()
