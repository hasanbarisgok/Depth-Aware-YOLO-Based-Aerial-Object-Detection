from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on a single image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("outputs/detection_result.jpg"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), conf=args.conf, verbose=False)
    result = results[0]

    plotted = result.plot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), plotted)

    print(f"Saved detection image to: {args.output}")
    boxes = result.obb if getattr(result, "obb", None) is not None else result.boxes
    if boxes is None:
        return

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        message = f"{model.names[cls_id]} | conf={conf:.4f} | bbox=[{x1}, {y1}, {x2}, {y2}]"
        if getattr(result, "obb", None) is not None and hasattr(box, "xyxyxyxy"):
            polygon = [[int(x), int(y)] for x, y in box.xyxyxyxy[0].tolist()]
            message += f" | polygon={polygon}"
        print(message)


if __name__ == "__main__":
    main()
