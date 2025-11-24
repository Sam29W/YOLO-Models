from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolo11n.pt")

print("🎯 YOLO Object Detection with Statistics Dashboard")
print("=" * 60)

# original detection
source = "https://ultralytics.com/images/bus.jpg"  # Change to your image
results = model.predict(source=source, conf=0.5, save=True)

# NEW ENHANCEMENT: Statistics Dashboard
for r in results:
    total_objects = len(r.boxes)
    print(f"\n✅ Total objects detected: {total_objects}")

    if total_objects == 0:
        print("No objects detected!")
        continue

    # Count each type
    object_counts = {}
    confidence_scores = {}

    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = r.names[class_id]
        confidence = float(box.conf[0])

        # Count objects
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

        # Track confidence scores
        if class_name not in confidence_scores:
            confidence_scores[class_name] = []
        confidence_scores[class_name].append(confidence)

    # to display Statistics Dashboard
    print("\n" + "=" * 60)
    print("📊 DETECTION STATISTICS DASHBOARD")
    print("=" * 60)

    print("\n📋 Object Distribution:")
    print("-" * 60)
    for obj_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_objects) * 100
        avg_confidence = sum(confidence_scores[obj_name]) / len(confidence_scores[obj_name])

        # it creates a simple bar chart
        bar = "█" * int(percentage / 5)

        print(f"{obj_name:15} | {count:2} ({percentage:5.1f}%) {bar}")
        print(f"{'':15} | Avg Confidence: {avg_confidence:.1%}")
        print()

    # Overall statistics
    print("-" * 60)
    all_confidences = [c for scores in confidence_scores.values() for c in scores]
    avg_overall = sum(all_confidences) / len(all_confidences)
    max_conf = max(all_confidences)
    min_conf = min(all_confidences)

    print(f"\n📈 Overall Statistics:")
    print(f"  • Average Confidence: {avg_overall:.1%}")
    print(f"  • Highest Confidence: {max_conf:.1%}")
    print(f"  • Lowest Confidence: {min_conf:.1%}")
    print(f"  • Most Common Object: {max(object_counts, key=object_counts.get)}")

    # Save statistics to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detection_stats_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("YOLO DETECTION STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Objects: {total_objects}\n\n")
        f.write("Object Distribution:\n")
        f.write("-" * 50 + "\n")
        for obj_name, count in sorted(object_counts.items()):
            percentage = (count / total_objects) * 100
            avg_conf = sum(confidence_scores[obj_name]) / len(confidence_scores[obj_name])
            f.write(f"{obj_name}: {count} ({percentage:.1f}%) - Avg Confidence: {avg_conf:.1%}\n")
        f.write("\n" + "-" * 50 + "\n")
        f.write(f"Average Overall Confidence: {avg_overall:.1%}\n")

    print(f"\n💾 Statistics saved to: {filename}")
    print("=" * 60)

print("\n✅ Detection complete with statistics!")
