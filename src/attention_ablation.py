"""Train and evaluate controlled ResNet50 attention ablations.

This closes the paper's missing ablation study by comparing four variants under
identical conditions:
    1) no attention
    2) channel attention only
    3) spatial attention only
    4) both attention blocks (current dual-attention design)

The study keeps the same data split, preprocessing, optimizer, loss function,
and training schedule while changing only the attention configuration of the
ResNet50 branch. The verified ensemble checkpoint remains untouched and is not
overwritten.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import AdamW

from dataset import CLASS_NAMES, get_dataloaders
from model import build_model


def sha256_file(path):
    hash_obj = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def evaluate_single_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float((all_preds == all_labels).mean())
    macro_precision = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    per_class_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": {CLASS_NAMES[i]: float(per_class_recall[i]) for i in range(len(CLASS_NAMES))},
        "per_class_precision": {CLASS_NAMES[i]: float(per_class_precision[i]) for i in range(len(CLASS_NAMES))},
        "per_class_f1": {CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(len(CLASS_NAMES))},
        "confusion_matrix": cm.tolist(),
    }


def configure_resnet_training(model, freeze_backbone=True):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "dual_attention"):
        for parameter in model.dual_attention.parameters():
            parameter.requires_grad = True

    for module in (model.head_conv, model.head_bn, model.fc1, model.fc2):
        for parameter in module.parameters():
            parameter.requires_grad = True

    if not freeze_backbone:
        for module in (model.layer2, model.layer3, model.layer4):
            for parameter in module.parameters():
                parameter.requires_grad = True


def train_single_variant(model, train_loader, val_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        configure_resnet_training(model, freeze_backbone=(epoch <= args.freeze_epochs))
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate_single_model(model, val_loader, device)
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_recall": val_metrics["macro_recall"],
                "attention_mode": args.attention_mode,
            }

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={running_loss / len(train_loader):.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_macro_recall={val_metrics['macro_recall']:.4f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError(f"No valid model state was produced for attention mode {args.attention_mode}")

    return best_state


def save_variant_summary(results, summary_path):
    rows = [
        "| Variant | Accuracy | Macro Recall | Macro Precision | Macro F1 | No_DR | Mild | Moderate | Severe | Proliferative_DR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for entry in results:
        m = entry["test_metrics"]
        rows.append(
            f"| {entry['attention_mode']} | {m['accuracy']:.4f} | {m['macro_recall']:.4f} | "
            f"{m['macro_precision']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['per_class_recall']['No_DR']:.4f} | {m['per_class_recall']['Mild']:.4f} | "
            f"{m['per_class_recall']['Moderate']:.4f} | {m['per_class_recall']['Severe']:.4f} | "
            f"{m['per_class_recall']['Proliferate_DR']:.4f} |"
        )

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


def run_ablation(args):
    if args.attention_mode is not None:
        attention_modes = [args.attention_mode]
    else:
        attention_modes = ["none", "channel", "spatial", "both"]

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    verified_checkpoint = os.path.join("checkpoints", "best_model_accuracy_RECOVERED.pth")
    if os.path.exists(verified_checkpoint):
        print(f"Verified checkpoint preserved: {verified_checkpoint}")
        print(f"Verified checkpoint SHA-256: {sha256_file(verified_checkpoint)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = []
    for attention_mode in attention_modes:
        print(f"\n=== Training attention mode: {attention_mode} ===")

        model = build_model(
            num_classes=5,
            feature_layer=args.feature_layer,
            pretrained=True,
            dropout=args.dropout,
            device=device,
            attention_mode=attention_mode,
        )

        if attention_mode == "none":
            checkpoint_name = "ablation_no_attention.pth"
        elif attention_mode == "channel":
            checkpoint_name = "ablation_channel_only.pth"
        elif attention_mode == "spatial":
            checkpoint_name = "ablation_spatial_only.pth"
        else:
            checkpoint_name = "ablation_dual_attention.pth"

        checkpoint_path = os.path.join("checkpoints", checkpoint_name)
        args_local = argparse.Namespace(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            freeze_epochs=args.freeze_epochs,
            attention_mode=attention_mode,
        )

        state = train_single_variant(model, train_loader, val_loader, device, args_local)
        model.load_state_dict({k: v.to(device) for k, v in state["model_state_dict"].items()})
        torch.save(state, checkpoint_path)

        test_metrics = evaluate_single_model(model, test_loader, device)
        result = {
            "attention_mode": attention_mode,
            "checkpoint": checkpoint_path,
            "epoch": state["epoch"],
            "val_accuracy": state["val_accuracy"],
            "val_macro_recall": state["val_macro_recall"],
            "test_metrics": test_metrics,
        }
        results.append(result)

        print(
            f"Mode={attention_mode} | test_accuracy={test_metrics['accuracy']:.4f} | "
            f"test_macro_recall={test_metrics['macro_recall']:.4f} | "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}",
            flush=True,
        )

    summary_json = os.path.join("outputs", "attention_ablation_results.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    summary_markdown = os.path.join("outputs", "attention_ablation_summary.md")
    save_variant_summary(results, summary_markdown)

    print(f"\nSaved JSON summary: {summary_json}")
    print(f"Saved markdown summary: {summary_markdown}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run controlled ResNet50 attention ablation study.")
    parser.add_argument("--attention_mode", type=str, choices=["none", "channel", "spatial", "both", "all"], default="all")
    parser.add_argument("--split_dir", type=str, default="data/split")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--feature_layer", type=str, default="layer3", choices=["layer2", "layer3"])
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if args.attention_mode == "all":
        args.attention_mode = None
    run_ablation(args)
