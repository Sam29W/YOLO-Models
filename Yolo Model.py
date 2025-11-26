from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolo11n.pt")

print("🎯 Smart Object Detection with Category Counter\n")
print("=" * 60)

# original model
source = "https://ultralytics.com/images/bus.jpg"  #add any image of your choice
results = model.predict(source=source, save=True)

# addition to the existing model
for r in results:
    total_objects = len(r.boxes)
    print(f"\n✅ Total objects detected: {total_objects}")

    #Counts each type of object
    object_counts = {}
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = r.names[class_id]
        confidence = float(box.conf[0])

        # counts only if confidence > 50%
        if confidence > 0.5:
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

    #Displays the breakdown
    print("\n📋 Object Breakdown:")
    print("-" * 30)
    for obj_name, count in sorted(object_counts.items()):
        print(f"  {obj_name}: {count}")
    print("-" * 30)

    #Saves results to text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_summary_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("YOLO Object Detection Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Objects: {total_objects}\n\n")
        f.write("Object Breakdown:\n")
        f.write("-" * 40 + "\n")
        for obj_name, count in sorted(object_counts.items()):
            f.write(f"  {obj_name}: {count}\n")

    print(f"\n💾 Results saved to: {filename}")

print("\n" + "=" * 60)
print("✅ Detection complete!")
