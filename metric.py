import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from dataset import get_dataloaders, CLASS_NAMES
from model import build_model


def main():
    checkpoint = os.path.join("checkpoints", "best_model.pth")
    split_dir = os.path.join("data", "split")
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No trained checkpoint found at '{checkpoint}'. Train the model first with: "
            "python src/train.py --epochs 30 --batch_size 32 --lr 1e-4"
        )

    os.makedirs("outputs", exist_ok=True)

    ckpt = torch.load(checkpoint, map_location=device)
    feature_layer = ckpt.get("feature_layer", "layer3")

    model = build_model(num_classes=5, feature_layer=feature_layer, pretrained=False, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, _, _, _ = get_dataloaders(split_dir=split_dir, batch_size=batch_size, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class = []
    for idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y_true == idx)
        support = int(mask.sum())
        if support == 0:
            p = r = f = 0.0
        else:
            p, r, f, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[idx], average=None, zero_division=0
            )
            p, r, f = float(p[0]), float(r[0]), float(f[0])
        per_class.append({
            "class": cls_name,
            "precision": p,
            "recall": r,
            "f1_score": f,
            "support": support,
        })

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("\n===== TOTAL MODEL METRICS =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-score: {f1_weighted:.4f}")

    print("\n===== PER-CLASS METRICS =====")
    for item in per_class:
        print(f"{item['class']:>15} | precision={item['precision']:.4f} | recall={item['recall']:.4f} | f1={item['f1_score']:.4f} | support={item['support']}")

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }

    with open(os.path.join("outputs", "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved outputs/model_metrics.json")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "confusion_matrix.png"), dpi=150)
    print("Saved outputs/confusion_matrix.png")

    recalls = [item["recall"] for item in per_class]
    plt.figure(figsize=(9, 5))
    plt.bar(CLASS_NAMES, recalls, color=["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#E91E63"])
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Per-Class Recall")
    plt.ylabel("Recall")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "per_class_recall.png"), dpi=150)
    print("Saved outputs/per_class_recall.png")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(per_class) + 1), recalls, marker="o", linestyle="-", color="royalblue")
    plt.xticks(range(1, len(per_class) + 1), CLASS_NAMES, rotation=20)
    plt.title("Recall by Class")
    plt.ylabel("Recall")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "recall_by_class.png"), dpi=150)
    print("Saved outputs/recall_by_class.png")

    print("\nAll evaluation artifacts saved in outputs/ folder.")


if __name__ == "__main__":
    main()
