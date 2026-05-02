# OBB_HA_HB Training Flow

This folder contains the YOLO OBB training workflow used for the current production checkpoint.

## Contents

```text
OBB_HA_HB/
+-- configs/
|   +-- data.yaml
+-- scripts/
    +-- prepare_dataset.py
    +-- train_yolo.py
```

Final trained weights are stored at the repository root under:

```text
models/obb_ha_hb_best.pt
models/obb_ha_hb_last.pt
```

Training artifacts are stored under:

```text
results/obb_ha_hb/metrics/
results/obb_ha_hb/samples/
```

## Prepare OBB Dataset

```powershell
python OBB_HA_HB/scripts/prepare_dataset.py
```

The script reads the original source folders:

```text
Images/
Annotations/YOLOv8 OBB format/
```

and prepares an OBB-ready dataset under:

```text
OBB_HA_HB/dataset/
```

The generated dataset folder is intentionally ignored by Git.

## Train

```powershell
python OBB_HA_HB/scripts/train_yolo.py
```

Default training parameters:

```text
task: obb
model: yolo26m-obb.pt
epochs: 100
imgsz: 640
batch: 64
name: OBB_HA_HB
patience: 15
```

## Final Recorded Metrics

| Metric | Value |
| --- | ---: |
| Precision | `0.97444` |
| Recall | `0.97254` |
| mAP50 | `0.98461` |
| mAP50-95 | `0.83686` |

Full logs and plots:

```text
results/obb_ha_hb/metrics/results.csv
results/obb_ha_hb/metrics/results.png
results/obb_ha_hb/metrics/confusion_matrix.png
results/obb_ha_hb/metrics/confusion_matrix_normalized.png
```
