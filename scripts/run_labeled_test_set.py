from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def write_temp_data_yaml(dataset_root: Path, output_dir: Path) -> Path:
    config_path = output_dir / "dataset_test_config.yaml"
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: airplane",
        "  1: bird",
        "  2: drone",
        "  3: helicopter",
    ]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def export_predictions_to_csv(results, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for result in results:
            image_name = Path(result.path).name
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                writer.writerow([image_name, cls_id, names[cls_id], f"{conf:.6f}", x1, y1, x2, y2])


def export_metrics(metrics, output_dir: Path) -> None:
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "fitness": float(metrics.fitness),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"precision: {summary['precision']:.6f}",
        f"recall: {summary['recall']:.6f}",
        f"mAP50: {summary['map50']:.6f}",
        f"mAP50-95: {summary['map50_95']:.6f}",
        f"fitness: {summary['fitness']:.6f}",
    ]
    (output_dir / "metrics_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all labeled test-set checks and save outputs.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predict_dir = args.output_dir / "annotated_predictions"
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    results = model.predict(
        source=str(args.images_dir),
        conf=args.conf,
        save=True,
        project=str(predict_dir),
        name="images",
        exist_ok=True,
        verbose=True,
    )
    export_predictions_to_csv(results, args.output_dir / "predictions.csv")

    data_yaml = write_temp_data_yaml(args.dataset_root, args.output_dir)
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(metrics_dir),
        name="evaluation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    export_metrics(metrics, args.output_dir)

    print(f"Annotated predictions saved under: {predict_dir}")
    print(f"Metrics saved under: {metrics_dir}")
    print(f"Prediction CSV saved to: {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
