from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # NEW: 16MB max file size

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO("yolo11n.pt")

# NEW: Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    # NEW: Better error handling
    try:
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded! Please select an image.")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected! Please choose an image.")

        # NEW: Check file type
        if not allowed_file(file.filename):
            return render_template('index.html',
                                   error="Invalid file type! Please upload PNG, JPG, JPEG, GIF, BMP, or WEBP.")

        # Start timer
        start_time = time.time()

        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"upload_{timestamp}.{file_extension}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get confidence from form
        confidence = float(request.form.get('confidence', 0.5))

        # Run detection
        results = model.predict(source=filepath, conf=confidence, save=False)

        # Process results
        detections = []
        object_counts = {}

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                conf = float(box.conf[0])

                detections.append({
                    'class': class_name,
                    'confidence': f"{conf:.2%}"
                })

                # Count objects
                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1

            # Save annotated image
            result_img = r.plot()
            result_filename = f"result_{timestamp}.jpg"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            Image.fromarray(result_img).save(result_path)

        # Calculate detection time
        detection_time = round(time.time() - start_time, 2)

        # NEW: Handle no detections
        if len(detections) == 0:
            return render_template('result.html',
                                   original=filename,
                                   result=result_filename,
                                   detections=[],
                                   total=0,
                                   object_counts={},
                                   detection_time=detection_time,
                                   no_objects=True)

        return render_template('result.html',
                               original=filename,
                               result=result_filename,
                               detections=detections,
                               total=len(detections),
                               object_counts=object_counts,
                               detection_time=detection_time,
                               no_objects=False)

    except Exception as e:
        # NEW: Catch any errors
        return render_template('index.html', error=f"An error occurred: {str(e)}")


@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['RESULT_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)


# NEW: Error handler for file too large
@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', error="File too large! Maximum size is 16MB."), 413


if __name__ == '__main__':
    import os

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
