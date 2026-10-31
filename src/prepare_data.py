"""
prepare_data.py
----------------
Splits the raw Kaggle "Diabetic Retinopathy 224x224 Gaussian Filtered" dataset
(class-per-folder format) into stratified train/val/test folders.

Expected raw layout (after unzipping the Kaggle download):
    data/raw/gaussian_filtered_images/
        No_DR/
        Mild/
        Moderate/
        Severe/
        Proliferate_DR/

Produces:
    data/split/train/<class>/*.png
    data/split/val/<class>/*.png
    data/split/test/<class>/*.png

Split ratio: 70% train / 15% val / 15% test, stratified per class so every
class keeps the same proportion in each split (important given the strong
class imbalance in this dataset).

Run once:
    python src/prepare_data.py
"""

import os
import shutil
import random
import argparse
import json

# Canonical label mapping used throughout the whole project.
CLASS_MAP = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
DISPLAY_NAMES = ["0 - No DR", "1 - Mild DR", "2 - Moderate DR", "3 - Severe DR", "4 - Proliferative DR"]


def split_dataset(raw_dir, out_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    random.seed(seed)
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    stats = {"train": {}, "val": {}, "test": {}}

    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {cls_dir}\n"
                f"Make sure the unzipped Kaggle dataset sits at: {raw_dir}"
            )
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            for fname in split_files:
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(out_dir, split, cls, fname)
                shutil.copyfile(src, dst)
            stats[split][cls] = len(split_files)

        print(f"[{cls:15s}] total={n:5d}  train={len(train_files):5d}  val={len(val_files):5d}  test={len(test_files):5d}")

    with open(os.path.join(out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSplit complete. Stats saved to", os.path.join(out_dir, "split_stats.json"))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw/gaussian_filtered_images",
                         help="Path to the unzipped Kaggle dataset (class-per-folder).")
    parser.add_argument("--out_dir", type=str, default="data/split",
                         help="Where to write train/val/test folders.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.raw_dir, args.out_dir, args.train_ratio, args.val_ratio, args.seed)
