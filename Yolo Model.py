from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.predict(source="")

for r in results:
    r.show()
