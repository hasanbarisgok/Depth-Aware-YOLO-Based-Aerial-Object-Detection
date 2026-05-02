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

Training source: Google Colab with an NVIDIA A100 GPU.

Run summary:

```text
Ultralytics: 8.4.46
Python: 3.12.13
PyTorch: 2.10.0+cu128
GPU: NVIDIA A100-SXM4-40GB
completed epochs: 86
training time: 5.450 hours
best epoch: 71
early stopping patience: 15
```

EarlyStopping stopped training after no improvement was observed for 15 consecutive epochs. The best validation results were observed at epoch 71 and saved as `best.pt`. Both `last.pt` and `best.pt` were optimizer-stripped to 48.1 MB.

Model summary:

```text
YOLO26m-obb fused model
layers: 142
parameters: 21,200,987
gradients: 0
GFLOPs: 71.5
```

| Metric | Value |
| --- | ---: |
| Precision | `0.976` |
| Recall | `0.972` |
| mAP50 | `0.987` |
| mAP50-95 | `0.83686` |

Per-class validation breakdown:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | `4514` | `6369` | `0.976` | `0.972` | `0.987` | `0.837` |
| airplane | `1119` | `1625` | `0.972` | `0.955` | `0.981` | `0.855` |
| bird | `638` | `1557` | `0.971` | `0.966` | `0.990` | `0.877` |
| drone | `1485` | `1602` | `0.978` | `0.981` | `0.983` | `0.804` |
| helicopter | `1158` | `1585` | `0.985` | `0.985` | `0.993` | `0.814` |

Validation speed:

| Stage | Time per image |
| --- | ---: |
| Preprocess | `0.1 ms` |
| Inference | `1.8 ms` |
| Loss | `0.0 ms` |
| Postprocess | `0.2 ms` |

Full logs and plots:

```text
results/obb_ha_hb/metrics/results.csv
results/obb_ha_hb/metrics/results.png
results/obb_ha_hb/metrics/confusion_matrix.png
results/obb_ha_hb/metrics/confusion_matrix_normalized.png
```

YOLO26m reference profile used for the final test context:

| Model | Image Size | mAP50-95 | mAP50 | CPU ONNX Latency | A100 TensorRT Latency | Params (M) | FLOPs (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO26m | `640` | `53.1` | `52.5` | `220.0 +/- 1.4 ms` | `4.7 +/- 0.1 ms` | `20.4` | `68.2` |
