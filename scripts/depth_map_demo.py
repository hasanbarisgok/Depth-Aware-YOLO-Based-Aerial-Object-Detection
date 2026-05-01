from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-depth map for a single image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--output", type=Path, default=Path("outputs/depth_map.png"))
    parser.add_argument("--comparison-output", type=Path, default=Path("outputs/depth_comparison.png"))
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    estimator = pipeline(task="depth-estimation", model=args.model)
    prediction = estimator(image)

    depth = np.array(prediction["depth"])
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), depth_colormap)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Pseudo-depth")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(args.comparison_output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved depth map to: {args.output}")
    print(f"Saved comparison image to: {args.comparison_output}")


if __name__ == "__main__":
    main()
