from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {
    0: "airplane",
    1: "bird",
    2: "drone",
    3: "helicopter",
}

SPLITS = ("train", "val", "test")


def count_files(path: Path, suffix: str | None = None) -> int:
    if not path.exists():
        return 0
    count = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and (suffix is None or entry.name.lower().endswith(suffix)):
                count += 1
    return count


def analyze_labels(label_dir: Path) -> tuple[Counter[int], Counter[int]]:
    images_with_class: Counter[int] = Counter()
    object_instances: Counter[int] = Counter()

    if not label_dir.exists():
        return images_with_class, object_instances

    with os.scandir(label_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".txt"):
                continue

            seen_in_image: set[int] = set()
            with open(entry.path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    object_instances[class_id] += 1
                    seen_in_image.add(class_id)

            for class_id in seen_in_image:
                images_with_class[class_id] += 1

    return images_with_class, object_instances


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    print(fmt(headers))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for row in rows:
        print(fmt(row))


def save_split_totals_chart(split_totals: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits = list(SPLITS)
    values = [split_totals[split] for split in splits]
    colors = ["#2563eb", "#16a34a", "#f97316"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(splits, values, color=colors)
    ax.set_title("Dataset Split Totals")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_distribution_chart(distribution: dict[str, dict[int, dict[str, int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = list(CLASS_NAMES)
    class_labels = [CLASS_NAMES[class_id] for class_id in class_ids]
    x = np.arange(len(class_ids))
    width = 0.24
    colors = {"train": "#2563eb", "val": "#16a34a", "test": "#f97316"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, split in enumerate(SPLITS):
        values = [distribution[split][class_id]["images"] for class_id in class_ids]
        offset = (index - 1) * width
        bars = ax.bar(x + offset, values, width, label=split, color=colors[split])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("Images With Class by Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_object_instances_chart(distribution: dict[str, dict[int, dict[str, int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_ids = list(CLASS_NAMES)
    class_labels = [CLASS_NAMES[class_id] for class_id in class_ids]
    x = np.arange(len(class_ids))
    width = 0.24
    colors = {"train": "#2563eb", "val": "#16a34a", "test": "#f97316"}

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, split in enumerate(SPLITS):
        values = [distribution[split][class_id]["instances"] for class_id in class_ids]
        offset = (index - 1) * width
        bars = ax.bar(x + offset, values, width, label=split, color=colors[split])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_title("Object Instances by Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Object Instance Count")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print train/val/test class distribution for a YOLO dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    split_totals: dict[str, int] = {}
    distribution: dict[str, dict[int, dict[str, int]]] = {}

    total_rows: list[list[str]] = []
    for split in SPLITS:
        image_count = count_files(dataset_root / "images" / split)
        label_count = count_files(dataset_root / "labels" / split, ".txt")
        split_totals[split] = image_count
        total_rows.append([split, str(image_count), str(label_count)])

    print("\nSplit totals")
    print_table(["split", "images", "labels"], total_rows)

    distribution_rows: list[list[str]] = []
    for split in SPLITS:
        images_with_class, object_instances = analyze_labels(dataset_root / "labels" / split)
        split_image_count = split_totals[split] or 1
        distribution[split] = {}

        for class_id, class_name in CLASS_NAMES.items():
            image_count = images_with_class[class_id]
            instance_count = object_instances[class_id]
            image_ratio = image_count / split_image_count * 100
            distribution[split][class_id] = {"images": image_count, "instances": instance_count}
            distribution_rows.append(
                [
                    split,
                    class_name,
                    str(image_count),
                    f"{image_ratio:.2f}%",
                    str(instance_count),
                ]
            )

    print("\nClass distribution")
    print_table(["split", "class", "images with class", "image ratio", "object instances"], distribution_rows)

    if args.output_dir is not None:
        save_split_totals_chart(split_totals, args.output_dir / "split_totals.png")
        save_class_distribution_chart(distribution, args.output_dir / "class_images_distribution.png")
        save_object_instances_chart(distribution, args.output_dir / "class_instances_distribution.png")
        print(f"\nCharts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
