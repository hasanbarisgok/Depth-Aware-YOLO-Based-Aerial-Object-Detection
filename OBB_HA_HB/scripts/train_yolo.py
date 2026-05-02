from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Train a YOLO26 OBB model on the prepared dataset.")
    parser.add_argument("--data", type=Path, default=project_root / "configs" / "data.yaml")
    parser.add_argument("--model", default="yolo26m-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="OBB_HA_HB")
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    model = YOLO(args.model, task="obb")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
