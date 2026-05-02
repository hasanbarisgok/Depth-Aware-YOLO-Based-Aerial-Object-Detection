# OBB_HA_HB Result Artifacts

This directory contains the curated artifacts from the final YOLO OBB training run.

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
| Precision | `0.97444` |
| Recall | `0.97254` |
| mAP50 | `0.98461` |
| mAP50-95 | `0.83686` |

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
