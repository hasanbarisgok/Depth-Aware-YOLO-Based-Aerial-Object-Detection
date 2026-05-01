from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid"),
    "test": ("test",),
}


@dataclass
class SplitStats:
    name: str
    widths: list[int]
    heights: list[int]
    sizes: list[tuple[int, int]]
    unreadable_files: int

    @property
    def total_images(self) -> int:
        return len(self.sizes)


def find_dataset_root(user_path: Path | None) -> Path:
    if user_path is not None:
        if not user_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {user_path}")
        return user_path.resolve()

    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    candidates = [
        repo_dir / "dataset" / "images",
        repo_dir.parent / "project" / "dataset" / "images",
        repo_dir.parent / "Images",
        Path.cwd() / "dataset" / "images",
        Path.cwd() / "Images",
        Path.cwd(),
    ]

    for candidate in candidates:
        if candidate.exists() and resolve_split_dirs(candidate):
            return candidate.resolve()

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not locate a supported dataset layout automatically.\n"
        "Checked the following paths:\n"
        f"{searched}\n"
        "Use --dataset-root to pass the dataset path explicitly."
    )


def resolve_split_dirs(dataset_root: Path) -> dict[str, Path]:
    layouts = [
        {split: dataset_root / split / "images" for split in ("train", "val", "test")},
        {split: dataset_root / split for split in ("train", "val", "test")},
        {"train": dataset_root / "train", "val": dataset_root / "valid", "test": dataset_root / "test"},
    ]

    for layout in layouts:
        if all(path.exists() and path.is_dir() for path in layout.values()):
            return layout

    alias_layout: dict[str, Path] = {}
    for split_name, aliases in SPLIT_ALIASES.items():
        found = None
        for alias in aliases:
            candidate = dataset_root / alias
            if candidate.exists() and candidate.is_dir():
                found = candidate
                break
            candidate = dataset_root / alias / "images"
            if candidate.exists() and candidate.is_dir():
                found = candidate
                break
        if found is None:
            return {}
        alias_layout[split_name] = found
    return alias_layout


def iter_image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def analyze_split(name: str, directory: Path) -> SplitStats:
    widths: list[int] = []
    heights: list[int] = []
    sizes: list[tuple[int, int]] = []
    unreadable_files = 0

    for image_path in iter_image_files(directory):
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            unreadable_files += 1
            continue

        widths.append(width)
        heights.append(height)
        sizes.append((width, height))

    return SplitStats(
        name=name,
        widths=widths,
        heights=heights,
        sizes=sizes,
        unreadable_files=unreadable_files,
    )


def print_split_report(stats: SplitStats) -> None:
    print(f"=== {stats.name.upper()} SPLIT ===")
    print(f"Images scanned       : {stats.total_images}")
    print(f"Unreadable files     : {stats.unreadable_files}")

    if stats.total_images == 0:
        print("No valid images found.\n")
        return

    avg_width = sum(stats.widths) / stats.total_images
    avg_height = sum(stats.heights) / stats.total_images
    top_sizes = Counter(stats.sizes).most_common(5)

    print(
        f"Width  -> min: {min(stats.widths)} px | max: {max(stats.widths)} px | avg: {avg_width:.2f} px"
    )
    print(
        f"Height -> min: {min(stats.heights)} px | max: {max(stats.heights)} px | avg: {avg_height:.2f} px"
    )
    print("Most common resolutions:")
    for (width, height), count in top_sizes:
        print(f"- {width}x{height}: {count}")
    print()


def print_overall_report(stats_by_split: list[SplitStats]) -> None:
    all_widths = [width for stats in stats_by_split for width in stats.widths]
    all_heights = [height for stats in stats_by_split for height in stats.heights]
    all_sizes = [size for stats in stats_by_split for size in stats.sizes]
    total_unreadable = sum(stats.unreadable_files for stats in stats_by_split)

    if not all_sizes:
        print("No valid images were found across the selected splits.")
        return

    total_images = len(all_sizes)
    avg_width = sum(all_widths) / total_images
    avg_height = sum(all_heights) / total_images

    print("=== OVERALL SUMMARY ===")
    print(f"Total images scanned : {total_images}")
    print(f"Unreadable files     : {total_unreadable}")
    print(f"Width  -> min: {min(all_widths)} px | max: {max(all_widths)} px | avg: {avg_width:.2f} px")
    print(f"Height -> min: {min(all_heights)} px | max: {max(all_heights)} px | avg: {avg_height:.2f} px")
    print("Most common resolutions:")
    for (width, height), count in Counter(all_sizes).most_common(10):
        print(f"- {width}x{height}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze image dimensions for train/val/test splits without loading the full image data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Dataset root path. Supported layouts: "
            "'dataset/images/{train,val,test}', 'Images/{train,valid,test}', or '{split}/images'."
        ),
    )
    args = parser.parse_args()

    dataset_root = find_dataset_root(args.dataset_root)
    split_dirs = resolve_split_dirs(dataset_root)

    if not split_dirs:
        raise RuntimeError(f"Unsupported dataset layout: {dataset_root}")

    print(f"Dataset root resolved to: {dataset_root}\n")

    split_stats = [analyze_split(name, path) for name, path in split_dirs.items()]
    for stats in split_stats:
        print_split_report(stats)

    print_overall_report(split_stats)


if __name__ == "__main__":
    main()
