import time
import csv
from pathlib import Path
from ultralytics import YOLO

# Load models
models = {
    "YOLOv5n": YOLO("yolov5n.pt"),
    "YOLOv5s": YOLO("yolov5s.pt")
}

# Folder with test images
image_dir = Path("test_images")
images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

if not images:
    print("❌ No images found in test_images/ folder.")
    exit()

# Results
results_summary = []
per_image_results = []  # ✅ Store per-image details

# Inference loop
for name, model in models.items():
    total_time = 0
    total_detections = 0
    unique_classes = set()

    print(f"\n🔍 Running inference for: {name}")

    for img_path in images:
        start = time.time()
        result = model(img_path)
        end = time.time()

        boxes = result[0].boxes
        class_ids = [int(c) for c in boxes.cls]  # Detected class IDs

        total_time += (end - start)
        total_detections += len(boxes)
        unique_classes.update(class_ids)

        # ✅ Save per-image results
        per_image_results.append({
            "Image": img_path.name,
            "Model": name,
            "Inference Time (s)": round(end - start, 4),
            "Detection Count": len(boxes),
            "Detected Classes": ", ".join(map(str, class_ids))
        })

    avg_time = total_time / len(images)

    # ✅ Save summary
    results_summary.append({
        "Model": name,
        "Avg Inference Time (s)": round(avg_time, 4),
        "Total Detections": total_detections,
        "Unique Classes Detected": len(unique_classes)
    })

# ✅ Print summary table
print("\n📊 Comparison Summary:")
print("{:<10} {:<25} {:<20} {:<25}".format("Model", "Avg Inference Time (s)", "Total Detections", "Unique Classes Detected"))
for r in results_summary:
    print("{:<10} {:<25} {:<20} {:<25}".format(
        r["Model"], r["Avg Inference Time (s)"], r["Total Detections"], r["Unique Classes Detected"]
    ))

# ✅ Save summary to CSV
with open("comparison_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Avg Inference Time (s)", "Total Detections", "Unique Classes Detected"])
    for r in results_summary:
        writer.writerow([
            r["Model"],
            r["Avg Inference Time (s)"],
            r["Total Detections"],
            r["Unique Classes Detected"]
        ])

# ✅ Save per-image log to CSV
with open("per_image_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Image", "Model", "Inference Time (s)", "Detection Count", "Detected Classes"])
    writer.writeheader()
    writer.writerows(per_image_results)

print("\n📁 Saved summary to comparison_results.csv")
print("📁 Saved per-image details to per_image_results.csv")
