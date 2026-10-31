"""
evaluate.py
-----------
Loads the best checkpoint, evaluates on the held-out test split, and produces:
    outputs/confusion_matrix.png
    outputs/classification_report.json  (incl. per-class recall/precision/F1)
    outputs/gradcam_examples.png        (one Grad-CAM overlay per class)

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders, get_eval_transforms, CLASS_NAMES, DRDataset
from model import build_ensemble_model
from gradcam import GradCAM
from PIL import Image


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs", exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    feature_layer = ckpt.get("feature_layer", "layer3")
    resnet_weight = ckpt.get("resnet_weight", 0.7)

    model = build_ensemble_model(num_classes=5, feature_layer=feature_layer,
                                 pretrained=False, resnet_weight=resnet_weight, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, _, _, test_ds = get_dataloaders(
        split_dir=args.split_dir, batch_size=args.batch_size, num_workers=2
    )

    all_preds, all_resnet_preds, all_efficientnet_preds, all_labels = [], [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            resnet_probs, efficientnet_probs = model.branch_probabilities(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            resnet_preds = resnet_probs.argmax(dim=1).cpu().numpy()
            efficientnet_preds = efficientnet_probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_resnet_preds.extend(resnet_preds.tolist())
            all_efficientnet_preds.extend(efficientnet_preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_resnet_preds = np.array(all_resnet_preds)
    all_efficientnet_preds = np.array(all_efficientnet_preds)
    all_labels = np.array(all_labels)

    print(f"ResNet50 accuracy:       {(all_resnet_preds == all_labels).mean():.4f}")
    print(f"EfficientNet-B0 accuracy: {(all_efficientnet_preds == all_labels).mean():.4f}")
    print(f"Ensemble accuracy:       {(all_preds == all_labels).mean():.4f}")

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES,
                                     output_dict=True, zero_division=0)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    with open("outputs/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved outputs/confusion_matrix.png")

    # Grad-CAM examples: one per class, taken from the test set
    cam = GradCAM(model)
    eval_tf = get_eval_transforms()

    fig, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(4 * len(CLASS_NAMES), 8))
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        example_path = None
        for path, label in test_ds.samples:
            if label == cls_idx:
                example_path = path
                break
        if example_path is None:
            continue

        pil_img = Image.open(example_path).convert("RGB")
        input_tensor = eval_tf(pil_img).unsqueeze(0)

        heatmap, pred_class, probs = cam.generate(input_tensor, class_idx=cls_idx)
        overlay = cam.overlay_on_image(heatmap, pil_img)

        axes[0, cls_idx].imshow(pil_img.resize((224, 224)))
        axes[0, cls_idx].set_title(f"True: {cls_name}")
        axes[0, cls_idx].axis("off")

        axes[1, cls_idx].imshow(overlay)
        axes[1, cls_idx].set_title(f"Grad-CAM (pred={CLASS_NAMES[pred_class]})")
        axes[1, cls_idx].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/gradcam_examples.png", dpi=150)
    plt.close()
    print("Saved outputs/gradcam_examples.png")

    print(f"\nTest accuracy: {report['accuracy']:.4f}")
    print("Per-class recall:")
    for cls in CLASS_NAMES:
        print(f"  {cls:15s}: {report[cls]['recall']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--split_dir", type=str, default="data/split")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    run_evaluation(args)
