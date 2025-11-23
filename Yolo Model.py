from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.predict(source="https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/dachshund.jpg")

for r in results:
    r.show()
