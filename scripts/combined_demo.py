from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

try:
    from scripts.depth_overlay import draw_depth_colored_detections, normalize_depth_map
except ImportError:
    from depth_overlay import draw_depth_colored_detections, normalize_depth_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection and pseudo-depth estimation together.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--output", type=Path, default=Path("outputs/combined_demo.png"))
    args = parser.parse_args()

    original_bgr = cv2.imread(str(args.image))
    if original_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    detector = YOLO(str(args.weights))
    detection = detector.predict(source=str(args.image), conf=args.conf, verbose=False)[0]

    estimator = pipeline(task="depth-estimation", model=args.depth_model)
    depth_prediction = estimator(Image.open(args.image).convert("RGB"))
    depth_normalized = normalize_depth_map(depth_prediction["depth"], original_bgr.shape[:2])
    detected_bgr = draw_depth_colored_detections(original_bgr, detection, depth_normalized, detector.names)
    detected_rgb = cv2.cvtColor(detected_bgr, cv2.COLOR_BGR2RGB)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    depth_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(detected_rgb)
    axes[1].set_title("YOLO Detection + pDepth Color")
    axes[2].imshow(depth_rgb)
    axes[2].set_title("Pseudo-depth")
    for axis in axes:
        axis.axis("off")
    plt.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined demo to: {args.output}")


if __name__ == "__main__":
    main()
