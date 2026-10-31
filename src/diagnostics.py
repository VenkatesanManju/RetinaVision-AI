"""Standalone diagnostics for Severe and Proliferate_DR behavior.

This script does not modify training code or checkpoints. It reports split counts,
confusion rows, validation correctness overlap across two checkpoints, augmented
image grids, oversampling diversity, and confidence by correct/incorrect outcome.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from dataset import (  # noqa: E402
    CLASS_NAMES,
    DRDataset,
    get_eval_transforms,
    get_train_transforms,
    resolve_split_path,
)
from model import build_ensemble_model  # noqa: E402

TARGET_CLASSES = ("Severe", "Proliferate_DR")
TARGET_LABELS = {CLASS_NAMES.index(name) for name in TARGET_CLASSES}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def split_counts(split_dir):
    print("\n1. Exact class counts")
    counts = {}
    for split in ("train", "val", "test"):
        root = resolve_split_path(split_dir, split)
        dataset = DRDataset(root)
        values = dict(zip(CLASS_NAMES, dataset.class_counts().astype(int).tolist()))
        counts[split] = values
        print(f"{split}: {values}")
        for name in TARGET_CLASSES:
            print(f"  {split} {name}: {values[name]}")
    return counts


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_ensemble_model(
        num_classes=5,
        feature_layer=checkpoint.get("feature_layer", "layer3"),
        pretrained=False,
        resnet_weight=checkpoint.get("resnet_weight", 0.7),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(checkpoint_path, split_dir, device, batch_size):
    model, checkpoint = load_model(checkpoint_path, device)
    val_root = resolve_split_path(split_dir, "val")
    dataset = DRDataset(val_root, transform=get_eval_transforms())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    records = []
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            ensemble_probs = model(images).exp()
            resnet_probs, efficientnet_probs = model.branch_probabilities(images)
            for row in range(len(labels)):
                true_label = int(labels[row])
                probs = ensemble_probs[row].cpu().numpy()
                predicted = int(probs.argmax())
                records.append({
                    "filename": dataset.samples[offset + row][0],
                    "true": true_label,
                    "pred": predicted,
                    "confidence": float(probs[predicted]),
                    "resnet_pred": int(resnet_probs[row].argmax().item()),
                    "efficientnet_pred": int(efficientnet_probs[row].argmax().item()),
                })
            offset += len(labels)

    return records, checkpoint, model


def confusion_rows(records, title):
    print(f"\n2. Confusion rows: {title}")
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for record in records:
        matrix[record["true"], record["pred"]] += 1
    for name in TARGET_CLASSES:
        label = CLASS_NAMES.index(name)
        row = matrix[label]
        details = ", ".join(f"predicted {CLASS_NAMES[i]}: {int(row[i])}" for i in range(len(CLASS_NAMES)))
        print(f"{name} images -> {details}")
    return matrix


def correct_target_files(records, title):
    print(f"\n3. Correct validation filenames: {title}")
    result = {}
    for name in TARGET_CLASSES:
        label = CLASS_NAMES.index(name)
        files = sorted(record["filename"] for record in records if record["true"] == label and record["pred"] == label)
        result[name] = files
        print(f"{name}: {len(files)} correct")
        for filename in files:
            print(f"  {filename}")
    return result


def compare_correct_files(first, second):
    if second is None:
        print("\n3. Cross-run comparison skipped: provide --checkpoint-previous.")
        return
    print("\n3. Cross-run correct-file overlap")
    for name in TARGET_CLASSES:
        overlap = sorted(set(first[name]) & set(second[name]))
        print(f"{name}: overlap={len(overlap)}")
        for filename in overlap:
            print(f"  {filename}")


def denormalize(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * STD + MEAN
    return np.clip(image, 0, 1)


def save_augmented_grid(split_dir, output_path, samples_per_class, seed):
    print("\n4. Creating augmented image grid")
    random.seed(seed)
    train_root = resolve_split_path(split_dir, "train")
    raw_dataset = DRDataset(train_root)
    transform = get_train_transforms(aggressive=True)
    selected = {name: [] for name in TARGET_CLASSES}
    for path, label in raw_dataset.samples:
        name = CLASS_NAMES[label]
        if name in selected and len(selected[name]) < samples_per_class:
            selected[name].append(path)

    figure, axes = plt.subplots(len(TARGET_CLASSES), samples_per_class, figsize=(3 * samples_per_class, 6))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(TARGET_CLASSES):
        for col, path in enumerate(selected[name]):
            image = transform(Image.open(path).convert("RGB"))
            axes[row, col].imshow(denormalize(image))
            axes[row, col].set_title(f"{name}\n{os.path.basename(path)}", fontsize=7)
            axes[row, col].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved augmented grid: {output_path}")


def oversampling_report(split_dir, multiplier, batch_size):
    print("\n5. Effective oversampling diversity")
    train_root = resolve_split_path(split_dir, "train")
    dataset = DRDataset(train_root)
    total_samples = len(dataset)
    samples_per_epoch = (total_samples // batch_size) * batch_size
    weights = {label: (multiplier if label in (3, 4) else 1.0) for label in range(len(CLASS_NAMES))}
    weight_sum = sum(weights[label] for _, label in dataset.samples)
    for name in TARGET_CLASSES:
        label = CLASS_NAMES.index(name)
        unique = sum(sample_label == label for _, sample_label in dataset.samples)
        expected = samples_per_epoch * (unique * multiplier) / weight_sum
        print(f"{name}: {unique} unique underlying images -> approximately {expected:.1f} sampled images/epoch")
    print(f"Total sampler draws: {samples_per_epoch} (drop_last with batch_size={batch_size})")


def confidence_report(records, title):
    print(f"\n6. Confidence for target true classes: {title}")
    for name in TARGET_CLASSES:
        label = CLASS_NAMES.index(name)
        correct = [r["confidence"] for r in records if r["true"] == label and r["pred"] == label]
        incorrect = [r["confidence"] for r in records if r["true"] == label and r["pred"] != label]
        def average(values):
            return f"{np.mean(values) * 100:.2f}%" if values else "n/a"
        print(f"{name}: correct_n={len(correct)}, avg_correct_confidence={average(correct)}, "
              f"incorrect_n={len(incorrect)}, avg_incorrect_confidence={average(incorrect)}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = split_counts(args.split_dir)

    latest_records, latest_checkpoint, _ = evaluate_checkpoint(
        args.checkpoint_latest, args.split_dir, device, args.batch_size
    )
    latest_matrix = confusion_rows(latest_records, args.checkpoint_latest)
    latest_correct = correct_target_files(latest_records, args.checkpoint_latest)

    previous_correct = None
    if args.checkpoint_previous:
        previous_records, _, _ = evaluate_checkpoint(
            args.checkpoint_previous, args.split_dir, device, args.batch_size
        )
        correct_target_files(previous_records, args.checkpoint_previous)
        previous_correct = correct_target_files(previous_records, args.checkpoint_previous)
    compare_correct_files(latest_correct, previous_correct)

    grid_path = os.path.join(args.output_dir, "minority_augmented_grid.png")
    save_augmented_grid(args.split_dir, grid_path, args.grid_samples, args.seed)
    oversampling_report(args.split_dir, args.minority_multiplier, args.batch_size)
    confidence_report(latest_records, args.checkpoint_latest)

    report = {
        "counts": counts,
        "latest_checkpoint": args.checkpoint_latest,
        "latest_confusion_matrix": latest_matrix.tolist(),
        "latest_correct_target_files": latest_correct,
        "device": str(device),
    }
    with open(os.path.join(args.output_dir, "minority_diagnostics.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nSaved raw diagnostic summary: {args.output_dir}/minority_diagnostics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="data/split")
    parser.add_argument("--checkpoint-latest", required=True)
    parser.add_argument("--checkpoint-previous")
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minority-multiplier", type=float, default=2.0)
    parser.add_argument("--grid-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
