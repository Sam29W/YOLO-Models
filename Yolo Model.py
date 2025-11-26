from ultralytics import YOLO
from datetime import datetime
import os

model = YOLO("yolo11n.pt")

print("=" * 60)
print("🎯 YOLO OBJECT DETECTION SYSTEM")
print("=" * 60)
print("\nChoose Detection Mode:")
print("1. Basic Image Detection")
print("2. Smart Object Counter")
print("3. Statistics Dashboard")
print("4. Batch Image Processing")
print("5. Confidence Control")
print("-" * 60)

mode = input("\nSelect mode (1-5): ") or "1"

#MODE 1: BASIC IMAGE DETECTION
if mode == "1":
    print("\n📸 Basic Image Detection")
    print("=" * 60)

    source = input("Enter image path: ")
    results = model.predict(source=source, save=True)

    for r in results:
        print(f"\n✅ Detected {len(r.boxes)} objects")

    print("\n✅ Detection complete!")

#MODE 2: SMART OBJECT COUNTER
elif mode == "2":
    print("\n🔢 Smart Object Counter")
    print("=" * 60)

    source = input("Enter image path: ")
    results = model.predict(source=source, conf=0.5, save=True)

    for r in results:
        total_objects = len(r.boxes)
        print(f"\n✅ Total objects detected: {total_objects}")

        if total_objects == 0:
            print("No objects detected!")
            continue

        # Count each type of object
        object_counts = {}
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            confidence = float(box.conf[0])

            # Only count if confidence > 50%
            if confidence > 0.5:
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

        # Display the breakdown
        print("\n📋 Object Breakdown:")
        print("-" * 30)
        for obj_name, count in sorted(object_counts.items()):
            print(f"  {obj_name}: {count}")
        print("-" * 30)

        # Save results to text file
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

    print("\n✅ Detection complete!")

#MODE 3: STATISTICS DASHBOARD
elif mode == "3":
    print("\n📊 Statistics Dashboard")
    print("=" * 60)

    source = input("Enter image path: ")
    results = model.predict(source=source, conf=0.5, save=True)

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

        # Display Statistics Dashboard
        print("\n" + "=" * 60)
        print("📊 DETECTION STATISTICS DASHBOARD")
        print("=" * 60)

        print("\n📋 Object Distribution:")
        print("-" * 60)
        for obj_name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_objects) * 100
            avg_confidence = sum(confidence_scores[obj_name]) / len(confidence_scores[obj_name])

            # Create a simple bar chart
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

#MODE 4: BATCH IMAGE PROCESSING
elif mode == "4":
    print("\n📸 Batch Image Processor")
    print("=" * 60)

    print("\nEnter image paths (one per line, press Enter twice when done):")
    image_paths = []
    while True:
        path = input()
        if path == "":
            break
        image_paths.append(path)

    if not image_paths:
        print("❌ No images provided!")
    else:
        print(f"\n🔄 Processing {len(image_paths)} images...")
        print("-" * 60)

        all_results = []
        total_objects_found = 0

        for i, source in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing image...")

            results = model.predict(source=source, conf=0.5, save=True)

            for r in results:
                num_objects = len(r.boxes)
                total_objects_found += num_objects

                object_types = {}
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = r.names[class_id]
                    object_types[class_name] = object_types.get(class_name, 0) + 1

                print(f"  ✅ Found {num_objects} objects")
                if object_types:
                    print(f"  📋 Objects: {', '.join([f'{count} {name}' for name, count in object_types.items()])}")

                all_results.append({
                    "image_number": i,
                    "total_objects": num_objects,
                    "object_breakdown": object_types
                })

        # Summary Report
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 60)

        print(f"\n✅ Processed: {len(image_paths)} images")
        print(f"🎯 Total objects detected: {total_objects_found}")
        print(f"📈 Average objects per image: {total_objects_found / len(image_paths):.1f}")

        all_object_types = {}
        for result in all_results:
            for obj_name, count in result["object_breakdown"].items():
                all_object_types[obj_name] = all_object_types.get(obj_name, 0) + count

        if all_object_types:
            print(f"\n📋 Object Types Detected Across All Images:")
            print("-" * 60)
            for obj_name, count in sorted(all_object_types.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * (count * 2)
                print(f"{obj_name:15} | {count:2} {bar}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"batch_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("BATCH IMAGE PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images Processed: {len(image_paths)}\n")
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
        print("\n" + "=" * 60)
        print("✅ Batch processing complete!")

#MODE 5: CONFIDENCE CONTROL
elif mode == "5":
    print("\n🎯 Confidence Control Detection")
    print("=" * 60)

    source = input("Enter image path: ")

    print("\n📊 Choose your confidence threshold:")
    print("  • 0.3 = Low (detects more, less accurate)")
    print("  • 0.5 = Medium (balanced)")
    print("  • 0.7 = High (detects less, very accurate)")

    confidence_level = float(input("\nEnter confidence (0.3-0.9): ") or 0.5)

    print(f"\n✅ Using {confidence_level:.0%} confidence threshold")
    print("-" * 60)

    results = model.predict(source=source, conf=confidence_level, save=True)

    for r in results:
        total_objects = len(r.boxes)

        if total_objects == 0:
            print("\n⚠️ No objects detected at this confidence level!")
            print("💡 Try lowering the confidence threshold")
        else:
            print(f"\n✅ Detected {total_objects} objects")

            print("\n📋 Detection Details:")
            print("-" * 60)

            object_list = []
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = r.names[class_id]
                confidence = float(box.conf[0])

                # Visual confidence bar
                bar_length = int(confidence * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)

                print(f"{class_name:15} | {confidence:.1%} {bar}")
                object_list.append(f"{class_name} ({confidence:.1%})")

            print("-" * 60)

            # Statistics
            confidences = [float(box.conf[0]) for box in r.boxes]
            avg_conf = sum(confidences) / len(confidences)
            max_conf = max(confidences)
            min_conf = min(confidences)

            print(f"\n📈 Confidence Statistics:")
            print(f"  • Average: {avg_conf:.1%}")
            print(f"  • Highest: {max_conf:.1%}")
            print(f"  • Lowest: {min_conf:.1%}")

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_conf_{confidence_level:.1f}_{timestamp}.txt"

            with open(filename, 'w') as f:
                f.write("YOLO DETECTION REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Confidence Threshold: {confidence_level:.1%}\n")
                f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Objects: {total_objects}\n\n")
                f.write("Objects Detected:\n")
                f.write("-" * 50 + "\n")
                for obj in object_list:
                    f.write(f"  • {obj}\n")
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"\nAverage Confidence: {avg_conf:.1%}\n")
                f.write(f"Highest Confidence: {max_conf:.1%}\n")
                f.write(f"Lowest Confidence: {min_conf:.1%}\n")

            print(f"\n💾 Report saved to: {filename}")

    print("\n" + "=" * 60)
    print("✅ Detection complete!")

else:
    print("\n❌ Invalid mode selected!")

print("\n" + "=" * 60)
