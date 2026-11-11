"""
Azhar EyeSight - Dataset Setup & Verification Utility
Helps verify dataset structure, organize raw downloaded retinal fundus archives,
and split data into train, val, and test subsets without patient leakage.
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import config
from utils.preprocessing import validate_dataset_structure, SUPPORTED_EXTENSIONS


def display_dataset_status(data_dir: Path = config.DATA_DIR) -> None:
    """Prints a formatted summary of images currently present in each dataset split."""
    report = validate_dataset_structure(data_dir=data_dir)

    print("\n" + "=" * 65)
    print(" AZHAR EYESIGHT – DATASET INTEGRITY & SPLIT AUDIT")
    print("=" * 65)
    print(f"Data Directory  : {data_dir.resolve()}")
    print(f"Structure Status: {'[VALID]' if report['is_valid'] else '[INCOMPLETE / EMPTY]'}")
    print(f"Total Images    : {report['total_images']}")
    print("-" * 65)

    for split in ["train", "val", "test"]:
        split_dict = report["splits"].get(split, {})
        split_total = sum(split_dict.values())
        print(f"\nSplit: {split.upper()} (Total: {split_total} images)")
        for cls in config.CLASSES:
            count = split_dict.get(cls, 0)
            status_symbol = "OK" if count > 0 else "EMPTY"
            print(f"  - {config.CLASS_LABELS.get(cls, cls):<22}: {count:>4} images  [{status_symbol}]")

    if report["empty_classes"]:
        print("\n" + "-" * 65)
        print("Empty category folders detected:")
        for emp in report["empty_classes"]:
            print(f"  ! {emp}")
        print("\nPlease place retinal fundus images in the above subdirectories before training.")

    print("=" * 65 + "\n")


def organize_and_split(
    source_dir: Path,
    target_dir: Path = config.DATA_DIR,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = config.SEED
) -> None:
    """
    Organizes a flat or class-structured raw retinal fundus dataset directory into
    train, val, and test splits according to specified ratios.
    """
    random.seed(seed)
    train_ratio, val_ratio, test_ratio = split_ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"

    print("\n" + "=" * 65)
    print(" AZHAR EYESIGHT – DATASET SPLIT & INGESTION")
    print("=" * 65)
    print(f"Source Folder : {source_dir.resolve()}")
    print(f"Target Folder : {target_dir.resolve()}")
    print(f"Split Ratios  : Train {train_ratio*100:.0f}% | Val {val_ratio*100:.0f}% | Test {test_ratio*100:.0f}%")
    print("-" * 65)

    for cls in config.CLASSES:
        # Check if source contains class subfolder or alias
        cls_candidates = [
            source_dir / cls,
            source_dir / cls.replace("_", " "),
            source_dir / cls.replace("_", ""),
            source_dir / config.CLASS_LABELS[cls].lower()
        ]
        
        found_source_cls = None
        for candidate in cls_candidates:
            if candidate.is_dir():
                found_source_cls = candidate
                break

        if found_source_cls is None:
            print(f"[Warning] No matching source folder found for class '{cls}'. Skipping.")
            continue

        images = [
            f for f in found_source_cls.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        print(f"Processing '{cls}': {n_total} total images -> {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

        for split_name, file_list in splits.items():
            dest_folder = target_dir / split_name / cls
            dest_folder.mkdir(parents=True, exist_ok=True)
            for f in file_list:
                shutil.copy2(f, dest_folder / f.name)

    print("\nDataset successfully organized and split!")
    display_dataset_status(target_dir)


def main():
    parser = argparse.ArgumentParser(description="Azhar EyeSight – Dataset Setup and Status Tool")
    parser.add_argument("--status", action="store_true", help="Display current dataset split status")
    parser.add_argument("--source", type=str, default=None, help="Path to raw source folder to organize and split")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test split ratio (default: 0.15)")

    args = parser.parse_args()

    if args.source:
        organize_and_split(
            source_dir=Path(args.source),
            target_dir=config.DATA_DIR,
            split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio)
        )
    else:
        display_dataset_status()


if __name__ == "__main__":
    main()
