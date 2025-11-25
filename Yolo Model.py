from ultralytics import YOLO
from datetime import datetime
import os

model = YOLO("yolo11n.pt")

print("📸 YOLO Batch Image Processor")
print("=" * 60)

# NEW FEATURE: Process multiple images at once
image_urls = [
    "https://ultralytics.com/images/bus.jpg",
    "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/dachshund.jpg",
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1"
]

# You can also use local images
# image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]

print(f"\n🔄 Processing {len(image_urls)} images...")
print("-" * 60)

# Store all results
all_results = []
total_objects_found = 0

for i, source in enumerate(image_urls, 1):
    print(f"\n[{i}/{len(image_urls)}] Processing image...")

    # Detect objects
    results = model.predict(source=source, conf=0.5, save=True)

    for r in results:
        num_objects = len(r.boxes)
        total_objects_found += num_objects

        # Count object types
        object_types = {}
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            object_types[class_name] = object_types.get(class_name, 0) + 1

        print(f"  ✅ Found {num_objects} objects")
        print(f"  📋 Objects: {', '.join([f'{count} {name}' for name, count in object_types.items()])}")

        # Store results
        all_results.append({
            "image_number": i,
            "total_objects": num_objects,
            "object_breakdown": object_types
        })

# Summary Report
print("\n" + "=" * 60)
print("📊 BATCH PROCESSING SUMMARY")
print("=" * 60)

print(f"\n✅ Processed: {len(image_urls)} images")
print(f"🎯 Total objects detected: {total_objects_found}")
print(f"📈 Average objects per image: {total_objects_found / len(image_urls):.1f}")

# Collect all unique object types
all_object_types = {}
for result in all_results:
    for obj_name, count in result["object_breakdown"].items():
        all_object_types[obj_name] = all_object_types.get(obj_name, 0) + count

print(f"\n📋 Object Types Detected Across All Images:")
print("-" * 60)
for obj_name, count in sorted(all_object_types.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * (count * 2)
    print(f"{obj_name:15} | {count:2} {bar}")

# Save detailed report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = f"batch_report_{timestamp}.txt"

with open(report_file, 'w') as f:
    f.write("BATCH IMAGE PROCESSING REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Images Processed: {len(image_urls)}\n")
    f.write(f"Total Objects: {total_objects_found}\n\n")

    f.write("Individual Image Results:\n")
    f.write("-" * 50 + "\n")
    for result in all_results:
        f.write(f"\nImage #{result['image_number']}:\n")
        f.write(f"  Objects: {result['total_objects']}\n")
        f.write(f"  Breakdown: {result['object_breakdown']}\n")

    f.write("\n" + "-" * 50 + "\n")
    f.write("\nOverall Object Distribution:\n")
    for obj_name, count in sorted(all_object_types.items()):
        f.write(f"  {obj_name}: {count}\n")

print(f"\n💾 Detailed report saved to: {report_file}")
print(f"📁 Detected images saved in: runs/detect/predict/")
print("\n" + "=" * 60)
print("✅ Batch processing complete!")
