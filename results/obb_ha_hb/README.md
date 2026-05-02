# OBB_HA_HB Result Artifacts

This directory contains the curated artifacts from the final YOLO OBB training run.

Run source: Google Colab with an NVIDIA A100 GPU.

Training stopped through Ultralytics EarlyStopping after 86 completed epochs. No improvement was observed for the final 15 epochs, and the best result was observed at epoch 71. The final exported checkpoints were optimizer-stripped to 48.1 MB each.

## Metrics

```text
metrics/
+-- args.yaml
+-- results.csv
+-- results.png
+-- BoxF1_curve.png
+-- BoxP_curve.png
+-- BoxPR_curve.png
+-- BoxR_curve.png
+-- confusion_matrix.png
+-- confusion_matrix_normalized.png
+-- labels.jpg
```

Final recorded validation metrics:

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

Runtime context:

```text
Ultralytics 8.4.46
Python 3.12.13
torch 2.10.0+cu128
NVIDIA A100-SXM4-40GB
YOLO26m-obb fused: 142 layers, 21,200,987 parameters, 71.5 GFLOPs
Speed: 0.1 ms preprocess, 1.8 ms inference, 0.0 ms loss, 0.2 ms postprocess per image
```

## Samples

```text
samples/
+-- train_batch0.jpg
+-- train_batch1.jpg
+-- train_batch2.jpg
+-- val_batch0_labels.jpg
+-- val_batch0_pred.jpg
+-- val_batch1_labels.jpg
+-- val_batch1_pred.jpg
+-- val_batch2_labels.jpg
+-- val_batch2_pred.jpg
```

The production model weights are stored under `models/`, not in this result directory.
