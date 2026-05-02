from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}

CLASS_NAMES = ["airplane", "bird", "drone", "helicopter"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def sync_split(src_dir: Path, dst_dir: Path) -> int:
    ensure_dir(dst_dir)
    count = 0
    for src_file in src_dir.iterdir():
        if src_file.is_file():
            link_or_copy(src_file, dst_dir / src_file.name)
            count += 1
    return count


def write_data_yaml(project_root: Path) -> None:
    config_path = project_root / "configs" / "data.yaml"
    ensure_dir(config_path.parent)
    lines = [
        "path: ../dataset",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO OBB dataset structure under OBB_HA_HB/dataset.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Root folder containing Images and Annotations.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset"

    total_images = 0
    total_labels = 0

    for src_split, dst_split in SPLIT_MAP.items():
        image_src = args.source_root / "Images" / src_split
        label_src = args.source_root / "Annotations" / "YOLOv8 OBB format" / src_split / "labels"

        image_dst = dataset_root / "images" / dst_split
        label_dst = dataset_root / "labels" / dst_split

        if not image_src.exists():
            raise FileNotFoundError(f"Image source not found: {image_src}")
        if not label_src.exists():
            raise FileNotFoundError(f"OBB label source not found: {label_src}")

        total_images += sync_split(image_src, image_dst)
        total_labels += sync_split(label_src, label_dst)

    write_data_yaml(project_root)

    print(f"OBB dataset ready: {dataset_root}")
    print(f"Images processed: {total_images}")
    print(f"OBB labels processed: {total_labels}")
    print(f"Config written: {project_root / 'configs' / 'data.yaml'}")


if __name__ == "__main__":
    main()
