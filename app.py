from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO("yolo11n.pt")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('home'))

    # Save uploaded image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}.jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Get confidence from form
    confidence = float(request.form.get('confidence', 0.5))

    # Run detection
    results = model.predict(source=filepath, conf=confidence, save=False)

    # Process results
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            conf = float(box.conf[0])
            detections.append({
                'class': class_name,
                'confidence': f"{conf:.2%}"
            })

        # Save annotated image
        result_img = r.plot()
        result_filename = f"result_{timestamp}.jpg"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        Image.fromarray(result_img).save(result_path)

    return render_template('result.html',
                           original=filename,
                           result=result_filename,
                           detections=detections,
                           total=len(detections))



if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
