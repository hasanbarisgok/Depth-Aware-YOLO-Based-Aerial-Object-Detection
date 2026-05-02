# Dataset Distribution Visualizations

This folder contains visual summaries of the dataset split and class distribution used during the project workflow.

## Files

```text
split_totals.png                 # train/validation/test image counts
class_images_distribution.png    # number of images containing each class by split
class_instances_distribution.png # object instance counts for each class by split
```

## Regeneration

From a prepared YOLO dataset:

```powershell
python scripts/analyze_split_distribution.py `
  --dataset-root "dataset" `
  --output-dir "results/dataset_distribution"
```

The full dataset is intentionally not committed to the repository. These plots are retained as lightweight dataset documentation.
