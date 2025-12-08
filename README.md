# My First YOLO Object Detection Project 
(https://huggingface.co/spaces/Samith29/yolo-object-detection) check out the website here

(https://object-detection-model-03p2.onrender.com)

Hey! This is my first attempt at building an object detection model using YOLO. It's pretty cool - it can spot everyday objects in images!

## What Does This Do?

I built this simple project to learn how object detection works. The model can recognize things you see every day like:

- People and animals (dogs, cats, birds)
- Vehicles (cars, bikes, trucks)
- Everyday stuff (phones, bottles, chairs, laptops)
- Food items (pizza, hot dogs, apples)
- Sports equipment (balls, frisbees, skateboards)

It finds these objects in images and draws boxes around them with labels. Pretty neat!

## Getting Started

If you want to try this yourself, here's what you need:

**Install Python packages:**
pip install ultralytics

text

**Run the code:**
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.predict(source="your_image.jpg")

for r in results:
r.show()


Just replace `"your_image.jpg"` with any image you want to test!

## How I Built This

I used YOLOv11, which is one of the newest and fastest object detection models out there. The "nano" version (yolo11n) is lightweight and perfect for learning. It's pre-trained, so I didn't have to train it myself - it already knows how to detect 80 different types of objects!

## What I Learned

- How to set up PyCharm for AI projects
- Working with pre-trained models
- Understanding how object detection actually works
- Getting comfortable with Python and ML libraries

## What's Next?

I'm planning to:
- Try it with my webcam for real-time detection
- Test it on videos
- Maybe train a custom model to detect specific things I care about

## Tech Stack

- Python 3
- YOLOv11 (Ultralytics)
- PyCharm IDE

## Final Thoughts

This was a fun weekend project! Object detection is way cooler than I expected. If you're learning AI/ML, definitely give YOLO a try - it's surprisingly easy to get started.

---

Built with curiosity and a lot of Stack Overflow searches :)

I'll be adding more enchancements everyday 
plz do check it out

#update - NOV 24

- Detection Statistics Dashboard: Shows object distribution with visual bars
- Displays confidence scores and percentages
- Saves detailed statistics to file

- # Update - Nov 25 
- Batch Image Processing: Process multiple images at once
- Generates comprehensive summary reports
- Saves all detected images automatically
- Visual statistics for object distribution

 YOLO OBJECT DETECTION SYSTEM

Choose Detection Mode:
1. Basic Image Detection
2. Smart Object Counter
3. Statistics Dashboard
4. Batch Image Processing
5. Confidence Control (NEW!)

Select mode (1-5):

# Update (Nov 27)
- Image Info Display (Mode 6): Shows image dimensions, file size, resolution, and object size analysis

- #Update (Nov 29)
- the UI is better now and can detect multiple objects in the image

-# update 30
 -   now you can preview your images and it shows detection time and number of objects

-# Update dec 5 
  -   now you can choose between models

# it's optimised for mobile phones now (dec 8)

