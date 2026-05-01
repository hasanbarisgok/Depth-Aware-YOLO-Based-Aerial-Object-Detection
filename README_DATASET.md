# Dataset Note

This repository does not include the full training dataset.

Original dataset source:

- https://data.mendeley.com/datasets/cd5z895tr2/1

Included:

- representative labeled test samples
- final model weights
- final metrics
- example prediction outputs

Not included:

- full train split
- full validation split
- full test split
- temporary cache files
- large intermediate training artifacts

Expected dataset structure for full training:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Class mapping:

- 0: airplane
- 1: bird
- 2: drone
- 3: helicopter
