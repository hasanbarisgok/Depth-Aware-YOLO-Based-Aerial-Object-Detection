from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_NAMES = {
    0: "airplane",
    1: "bird",
    2: "drone",
    3: "helicopter",
}


def find_one_sample_per_class(labels_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label_file in sorted(labels_dir.glob("*.txt")):
        lines = label_file.read_text(encoding="utf-8").strip().splitlines()
        classes = {int(line.split()[0]) for line in lines if line.strip()}
        for cls_id, cls_name in CLASS_NAMES.items():
            if cls_id in classes and cls_name not in found:
                found[cls_name] = label_file
        if len(found) == len(CLASS_NAMES):
            break
    return found


def save_detection(model: YOLO, image_path: Path, output_path: Path, conf: float) -> list[list[str]]:
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    result = results[0]
    plotted = result.plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), plotted)

    rows: list[list[str]] = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        rows.append([image_path.name, CLASS_NAMES.get(cls_id, str(cls_id)), f"{score:.6f}", x1, y1, x2, y2])
    return rows


def write_readme(output_dir: Path) -> None:
    text = """# Completed Tests

This folder contains one selected labeled sample image for each class from the test split, together with the model outputs.

Each class folder contains:

- `input.jpg`: original test image
- `label.txt`: corresponding YOLO label file
- `prediction.jpg`: model prediction with rendered bounding boxes

Additionally, at the root level:

- `summary.csv`: prediction summary for all selected test samples
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one labeled sample test for each class.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_readme(args.output_dir)

    samples = find_one_sample_per_class(args.labels_dir)
    if len(samples) != len(CLASS_NAMES):
        missing = sorted(set(CLASS_NAMES.values()) - set(samples))
        raise RuntimeError(f"Missing sample labels for classes: {missing}")

    model = YOLO(str(args.weights))
    summary_rows: list[list[str]] = []

    for class_name, label_path in samples.items():
        image_path = args.images_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for label {label_path.name}: {image_path}")

        class_dir = args.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, class_dir / "input.jpg")
        shutil.copy2(label_path, class_dir / "label.txt")

        rows = save_detection(model, image_path, class_dir / "prediction.jpg", args.conf)
        if rows:
            summary_rows.extend(rows)
        else:
            summary_rows.append([image_path.name, "no_detection", "0.000000", "", "", "", ""])

    summary_csv = args.output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "predicted_class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(summary_rows)

    print(f"Sample tests saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
