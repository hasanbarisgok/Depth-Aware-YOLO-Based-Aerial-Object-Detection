# YOLOv8 50-Epoch Baseline Results

This folder preserves the earlier axis-aligned YOLOv8 baseline run. These artifacts predate the current YOLO OBB production model and do not include oriented bounding box support.

## Scope

```text
model family: YOLOv8
task: standard object detection
bounding boxes: axis-aligned
target epochs: 50
run name: aod4_total50_from13
```

## Files

```text
aod4_total50_results.csv   # epoch-by-epoch training and validation metrics
aod4_total50_status.txt    # run status and source run directory
FINAL_METRICS.md           # final validation summary for the baseline
sample_detection.jpg       # sample rendered baseline detection
```

## Final Baseline Metrics

| Metric | Value |
| --- | ---: |
| Precision | `0.94087` |
| Recall | `0.93935` |
| mAP50 | `0.96537` |
| mAP50-95 | `0.65577` |

Per-class summary:

| Class | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| airplane | `0.968` | `0.696` |
| bird | `0.973` | `0.694` |
| drone | `0.954` | `0.612` |
| helicopter | `0.968` | `0.622` |

## Current Model

The current production model is the YOLO OBB checkpoint stored at:

```text
models/obb_ha_hb_best.pt
```

The current OBB training artifacts are stored separately under:

```text
results/obb_ha_hb/
```
