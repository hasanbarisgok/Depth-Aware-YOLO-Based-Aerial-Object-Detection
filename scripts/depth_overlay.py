from __future__ import annotations

from typing import Any

import cv2
import numpy as np


DEPTH_COLORS_BGR = {
    "low": (255, 120, 0),
    "mid": (0, 210, 255),
    "high": (0, 60, 255),
}


def normalize_depth_map(depth: Any, target_shape: tuple[int, int]) -> np.ndarray:
    depth_array = np.array(depth)
    if depth_array.ndim == 3:
        depth_array = depth_array[:, :, 0]

    height, width = target_shape
    if depth_array.shape[:2] != (height, width):
        depth_array = cv2.resize(depth_array, (width, height), interpolation=cv2.INTER_CUBIC)

    normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def color_for_depth(mean_depth: float, low_threshold: float, high_threshold: float) -> tuple[int, int, int]:
    if mean_depth >= high_threshold:
        return DEPTH_COLORS_BGR["high"]
    if mean_depth >= low_threshold:
        return DEPTH_COLORS_BGR["mid"]
    return DEPTH_COLORS_BGR["low"]


def draw_depth_colored_detections(
    image_bgr: np.ndarray,
    result: Any,
    depth_normalized: np.ndarray,
    class_names: dict[int, str],
) -> np.ndarray:
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    depth_height, depth_width = depth_normalized.shape[:2]
    if (depth_height, depth_width) != (height, width):
        depth_normalized = cv2.resize(depth_normalized, (width, height), interpolation=cv2.INTER_CUBIC)

    low_threshold, high_threshold = np.percentile(depth_normalized, [33, 66])
    thickness = max(2, round(min(height, width) / 320))
    font_scale = max(0.45, min(height, width) / 1150)
    label_thickness = max(1, thickness - 1)

    obb = getattr(result, "obb", None)
    boxes = obb if obb is not None else result.boxes
    if boxes is None:
        return annotated

    for box in boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = depth_normalized[y1:y2, x1:x2]
        color = color_for_depth(float(crop.mean()), float(low_threshold), float(high_threshold))

        if obb is not None and hasattr(box, "xyxyxyxy"):
            points = np.array(box.xyxyxyxy[0].tolist(), dtype=np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=thickness)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = class_names.get(class_id, str(class_id))
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
        label_height = label_size[1] + baseline + 8
        label_width = label_size[0] + 10
        label_y1 = max(0, y1 - label_height)
        label_y2 = label_y1 + label_height
        label_x2 = min(width - 1, x1 + label_width)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, label_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )

    return annotated
