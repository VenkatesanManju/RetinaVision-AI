"""Compare accuracy and kappa checkpoints and render target-class Grad-CAMs."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from dataset import CLASS_NAMES, DRDataset, get_eval_transforms, resolve_split_path
from gradcam import GradCAM
from model import build_ensemble_model


TARGET_LABELS = {CLASS_NAMES.index("Severe"), CLASS_NAMES.index("Proliferate_DR")}


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_ensemble_model(
        num_classes=len(CLASS_NAMES),
        feature_layer=checkpoint.get("feature_layer", "layer3"),
        pretrained=False,
        resnet_weight=checkpoint.get("resnet_weight", 0.7),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def collect_predictions(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            log_probs = model(images.to(device))
            probabilities = log_probs.exp().cpu().numpy()
            for row, label in enumerate(labels.numpy()):
                predicted = int(probabilities[row].argmax())
                predictions.append({
                    "path": dataset.samples[len(predictions)][0],
                    "true": int(label),
                    "pred": predicted,
                    "confidence": float(probabilities[row, predicted]),
                })
    return predictions


def render_grid(records, model, output_path, device):
    transform = get_eval_transforms()
    cam = GradCAM(model)
    columns = 5
    rows = int(np.ceil(len(records) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, record in zip(axes, records):
        image = Image.open(record["path"]).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        heatmap, _, probabilities = cam.generate(tensor, class_idx=record["pred"])
        overlay = cam.overlay_on_image(heatmap, image)
        axis.imshow(overlay)
        axis.set_title(
            f"True: {CLASS_NAMES[record['true']]}\n"
            f"Pred: {CLASS_NAMES[record['pred']]} ({record['confidence'] * 100:.1f}%)"
        )
        axis.axis("off")
    for axis in axes[len(records):]:
        axis.axis("off")
    figure.suptitle("Kappa checkpoint Grad-CAM: newly correct and remaining errors", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_path = resolve_split_path(args.split_dir, "test")
    dataset = DRDataset(val_path, transform=get_eval_transforms())
    kappa_model = load_checkpoint(args.kappa_checkpoint, device)
    kappa_records = collect_predictions(kappa_model, dataset, args.batch_size, device)
    target_records = [record for record in kappa_records if record["true"] in TARGET_LABELS]

    newly_correct = []
    if args.previous_checkpoint and os.path.exists(args.previous_checkpoint):
        previous_model = load_checkpoint(args.previous_checkpoint, device)
        previous_records = collect_predictions(previous_model, dataset, args.batch_size, device)
        newly_correct = [
            current for current, previous in zip(kappa_records, previous_records)
            if current["true"] in TARGET_LABELS
            and current["pred"] == current["true"]
            and previous["pred"] != previous["true"]
        ]
        print(f"Newly correct target images: {len(newly_correct)}")
    else:
        print("Previous checkpoint unavailable; using current correct target images instead.")
        newly_correct = [record for record in target_records if record["pred"] == record["true"]]

    still_incorrect = [record for record in target_records if record["pred"] != record["true"]]
    selected = newly_correct[:args.newly_correct] + still_incorrect[:args.still_incorrect]
    render_grid(selected, kappa_model, args.output, device)
    print(f"Selected {len(selected)} images for Grad-CAM.")
    print(f"Saved Grad-CAM grid: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="data/split")
    parser.add_argument("--kappa-checkpoint", required=True)
    parser.add_argument("--previous-checkpoint")
    parser.add_argument("--output", default="outputs/gradcam_kappa_checkpoint.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--newly-correct", type=int, default=8)
    parser.add_argument("--still-incorrect", type=int, default=5)
    main(parser.parse_args())