"""
train.py
--------
Trains the Dual Attention-Enhanced ResNet50 DR classifier.

Usage (Kaggle notebook or local machine with GPU):
    python src/train.py --epochs 30 --batch_size 32 --lr 1e-4

Saves:
    checkpoints/best_model.pth   (best validation macro-recall checkpoint)
    outputs/training_history.json
    outputs/training_curves.png
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, cohen_kappa_score
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders, CLASS_NAMES
from model import build_ensemble_model
from losses import compute_class_weights, OrdinalCORALLoss


def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_preds, all_resnet_preds, all_efficientnet_preds, all_labels = [], [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            resnet_probs, efficientnet_probs = model.branch_probabilities(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()
                n_batches += 1
            preds = logits.argmax(dim=1)
            resnet_preds = resnet_probs.argmax(dim=1)
            efficientnet_preds = efficientnet_probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_resnet_preds.extend(resnet_preds.cpu().numpy().tolist())
            all_efficientnet_preds.extend(efficientnet_preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_resnet_preds = np.array(all_resnet_preds)
    all_efficientnet_preds = np.array(all_efficientnet_preds)
    all_labels = np.array(all_labels)

    per_class_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0)
    resnet_per_class_recall = recall_score(
        all_labels, all_resnet_preds, average=None, labels=list(range(len(CLASS_NAMES)),), zero_division=0
    )
    efficientnet_per_class_recall = recall_score(
        all_labels, all_efficientnet_preds, average=None, labels=list(range(len(CLASS_NAMES)),), zero_division=0
    )
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    resnet_kappa = cohen_kappa_score(all_labels, all_resnet_preds, weights="quadratic")
    efficientnet_kappa = cohen_kappa_score(all_labels, all_efficientnet_preds, weights="quadratic")
    accuracy = (all_preds == all_labels).mean()
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    metrics = {
        "loss": total_loss / max(n_batches, 1) if criterion is not None else None,
        "accuracy": float(accuracy),
        "resnet_accuracy": float((all_resnet_preds == all_labels).mean()),
        "efficientnet_accuracy": float((all_efficientnet_preds == all_labels).mean()),
        "macro_recall": float(macro_recall),
        "macro_precision": float(macro_precision),
        "macro_f1": float(macro_f1),
        "weighted_kappa": float(weighted_kappa),
        "per_class_recall": {CLASS_NAMES[i]: float(per_class_recall[i]) for i in range(len(CLASS_NAMES))},
        "resnet_macro_recall": float(recall_score(all_labels, all_resnet_preds, average="macro", zero_division=0)),
        "efficientnet_macro_recall": float(recall_score(all_labels, all_efficientnet_preds, average="macro", zero_division=0)),
        "resnet_weighted_kappa": float(resnet_kappa),
        "efficientnet_weighted_kappa": float(efficientnet_kappa),
        "resnet_per_class_recall": {
            CLASS_NAMES[i]: float(resnet_per_class_recall[i]) for i in range(len(CLASS_NAMES))
        },
        "efficientnet_per_class_recall": {
            CLASS_NAMES[i]: float(efficientnet_per_class_recall[i]) for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def apply_restricted_mixup(images, labels, alpha=0.4):
    """Mix only pairs from the Moderate/Severe/Proliferate_DR zone."""
    eligible = torch.where((labels >= 2) & (labels <= 4))[0]
    if alpha <= 0 or eligible.numel() < 2:
        return images, labels, labels, 1.0

    shuffled = eligible[torch.randperm(eligible.numel(), device=labels.device)]
    permutation = torch.arange(labels.numel(), device=labels.device)
    permutation[eligible] = shuffled
    lam = float(np.random.beta(alpha, alpha))
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    return mixed_images, labels, labels[permutation], lam


def configure_fine_tuning(model, epoch=1, freeze_epochs=0):
    """Train classification heads first, then unfreeze the final backbone blocks."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    head_modules = (model.resnet.dual_attention, model.resnet.head_conv,
                    model.resnet.head_bn, model.resnet.fc1, model.resnet.fc2,
                    model.efficientnet.classifier)
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if epoch <= freeze_epochs:
        return

    for module in (model.resnet.layer2, model.resnet.layer3, model.resnet.layer4):
        for parameter in module.parameters():
            parameter.requires_grad = True
    feature_blocks = list(model.efficientnet.features.children())
    for module in feature_blocks[-3:]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def configure_focus_stage(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.resnet.dual_attention, model.resnet.head_conv,
                   model.resnet.head_bn, model.resnet.fc1, model.resnet.fc2,
                   model.efficientnet.classifier):
        for parameter in module.parameters():
            parameter.requires_grad = True


def write_comparison_report(test_metrics):
    baseline_path = "outputs/model_metrics.json"
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as handle:
            baseline = json.load(handle)
    baseline_per_class = {
        row["class"]: row["recall"] for row in baseline.get("per_class", [])
    }
    after_per_class = test_metrics["per_class_recall"]
    headers = ["Model", "Phase", *CLASS_NAMES, "Macro recall", "Macro F1", "Weighted kappa", "Accuracy"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    before_values = [baseline_per_class.get(name, "n/a") for name in CLASS_NAMES]
    before_values.extend([
        baseline.get("macro_recall", "n/a"), baseline.get("macro_f1", "n/a"),
        "n/a", baseline.get("accuracy", "n/a")
    ])
    lines.append("| Ensemble | Before | " + " | ".join(str(value) for value in before_values) + " |")
    after_values = [after_per_class[name] for name in CLASS_NAMES]
    after_values.extend([
        test_metrics["macro_recall"], test_metrics["macro_f1"],
        test_metrics["weighted_kappa"], test_metrics["accuracy"]
    ])
    lines.append("| Ensemble | After | " + " | ".join(f"{value:.4f}" for value in after_values) + " |")
    for branch, prefix in (("ResNet50", "resnet"), ("EfficientNet-B0", "efficientnet")):
        branch_values = test_metrics[f"{prefix}_per_class_recall"]
        branch_values = [branch_values[name] for name in CLASS_NAMES]
        branch_values.extend([
            test_metrics[f"{prefix}_macro_recall"], "n/a",
            test_metrics[f"{prefix}_weighted_kappa"],
            test_metrics[f"{prefix}_accuracy"],
        ])
        lines.append("| " + branch + " | After | " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in branch_values) + " |")
    with open("outputs/metrics_comparison.md", "w") as handle:
        handle.write("\n".join(lines) + "\n")


def train(args):
    if not torch.cuda.is_available():
        cpus = os.cpu_count() or 4
        torch.set_num_threads(cpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        split_dir=args.split_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        oversample_minority=args.oversample_minority,
        minority_multiplier=args.minority_multiplier,
    )

    print("Train class counts:", dict(zip(CLASS_NAMES, train_ds.class_counts())), flush=True)

    if args.objective == "accuracy":
        criterion = nn.CrossEntropyLoss()
        print("Objective: validation accuracy (unweighted cross-entropy)", flush=True)
    elif args.objective == "weighted_ce":
        class_weights = compute_class_weights(
            train_ds.class_counts(), mode="inverse"
        )
        print("Objective: inverse-frequency weighted cross-entropy", flush=True)
        print("Class weights:", class_weights.tolist(), flush=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        class_weights = compute_class_weights(
            train_ds.class_counts(), mode="inverse"
        )
        print("Objective: inverse-frequency ordinal CORAL loss", flush=True)
        print("Ordinal class weights:", class_weights.tolist(), flush=True)
        criterion = OrdinalCORALLoss(num_classes=len(CLASS_NAMES), class_weights=class_weights)

    model = build_ensemble_model(num_classes=5, feature_layer=args.feature_layer,
                                 pretrained=True, dropout=args.dropout,
                                 resnet_weight=args.resnet_weight, device=device)

    configure_fine_tuning(model, epoch=1, freeze_epochs=args.freeze_epochs)
    print(f"Training schedule: backbone frozen for {args.freeze_epochs} epoch(s), then fine-tuned.", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_resnet_accuracy": [],
               "val_efficientnet_accuracy": [], "val_macro_recall": [],
               "val_per_class_recall": []}

    best_metric = -1.0
    best_focus_recall = -1.0
    best_kappa = -1.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        configure_fine_tuning(model, epoch=epoch, freeze_epochs=args.freeze_epochs)
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            mixed_images, labels_a, labels_b, lam = apply_restricted_mixup(
                images, labels, alpha=args.mixup_alpha
            )
            optimizer.zero_grad()
            logits = model(mixed_images)
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                print(f"   Batch [{step:02d}/{len(train_loader):02d}] loss={loss.item():.4f}", flush=True)

        train_loss = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device, criterion)

        selection_metric = val_metrics[args.selection_metric]
        scheduler.step(selection_metric)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_resnet_accuracy"].append(val_metrics["resnet_accuracy"])
        history["val_efficientnet_accuracy"].append(val_metrics["efficientnet_accuracy"])
        history["val_macro_recall"].append(val_metrics["macro_recall"])
        history["val_per_class_recall"].append(val_metrics["per_class_recall"])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"resnet_acc={val_metrics['resnet_accuracy']:.4f} | "
              f"efficientnet_acc={val_metrics['efficientnet_accuracy']:.4f} | "
              f"val_macro_recall={val_metrics['macro_recall']:.4f} | {elapsed:.1f}s", flush=True)
        print("   per-class recall:", {k: round(v, 3) for k, v in val_metrics["per_class_recall"].items()}, flush=True)

        focus_metric = (
            val_metrics["per_class_recall"][CLASS_NAMES[1]]
            + val_metrics["per_class_recall"][CLASS_NAMES[2]]
            + val_metrics["per_class_recall"][CLASS_NAMES[4]]
        ) / 3.0

        # Model selection emphasizes the clinically important classes while still
        # maintaining overall macro recall. This reduces the Mild/Moderate and
        # Proliferative/Moderate misclassification pattern without overfitting to
        # the majority class.
        if selection_metric > best_metric or (
            abs(selection_metric - best_metric) < 1e-4 and focus_metric > best_focus_recall
        ):
            best_metric = selection_metric
            best_focus_recall = focus_metric
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": "resnet50_efficientnet_b0_ensemble",
                "objective": args.objective,
                "selection_metric": args.selection_metric,
                "resnet_weight": args.resnet_weight,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "feature_layer": args.feature_layer,
            }, args.checkpoint_path)
            print(f"   -> New best model saved ({args.selection_metric}={selection_metric:.4f}, "
                f"macro_recall={val_metrics['macro_recall']:.4f}, focus_recall={focus_metric:.4f})", flush=True)
        else:
            epochs_no_improve += 1

        if args.save_kappa_checkpoint and val_metrics["weighted_kappa"] > best_kappa:
            best_kappa = val_metrics["weighted_kappa"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": "resnet50_efficientnet_b0_ensemble",
                "objective": args.objective,
                "selection_metric": "weighted_kappa",
                "resnet_weight": args.resnet_weight,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "feature_layer": args.feature_layer,
            }, args.kappa_checkpoint_path)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.", flush=True)
            break

    if args.second_stage_epochs > 0:
        focus_indices = [index for index, (_, label) in enumerate(train_ds.samples) if 2 <= label <= 4]
        focus_loader = DataLoader(
            Subset(train_ds, focus_indices), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
        )
        configure_focus_stage(model)
        focus_optimizer = AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=args.lr * args.second_stage_lr_scale, weight_decay=args.weight_decay
        )
        print(f"\nSecond focus stage: {len(focus_indices)} Moderate/Severe/Proliferate_DR images for "
              f"{args.second_stage_epochs} epochs.", flush=True)
        for stage_epoch in range(1, args.second_stage_epochs + 1):
            model.train()
            for images, labels in focus_loader:
                images, labels = images.to(device), labels.to(device)
                mixed_images, labels_a, labels_b, lam = apply_restricted_mixup(
                    images, labels, alpha=args.mixup_alpha
                )
                focus_optimizer.zero_grad()
                logits = model(mixed_images)
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                focus_optimizer.step()
            focus_metrics = evaluate(model, val_loader, device, criterion)
            print(f"Focus epoch {stage_epoch:02d}/{args.second_stage_epochs} | "
                  f"val_macro_recall={focus_metrics['macro_recall']:.4f} | "
                  f"val_weighted_kappa={focus_metrics['weighted_kappa']:.4f}", flush=True)
            if focus_metrics["macro_recall"] > best_metric:
                best_metric = focus_metrics["macro_recall"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_type": "resnet50_efficientnet_b0_ensemble",
                    "objective": args.objective,
                    "selection_metric": "macro_recall",
                    "resnet_weight": args.resnet_weight,
                    "epoch": args.epochs + stage_epoch,
                    "val_metrics": focus_metrics,
                    "feature_layer": args.feature_layer,
                }, "checkpoints/best_model.pth")
            if focus_metrics["weighted_kappa"] > best_kappa:
                best_kappa = focus_metrics["weighted_kappa"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_type": "resnet50_efficientnet_b0_ensemble",
                    "objective": args.objective,
                    "selection_metric": "weighted_kappa",
                    "resnet_weight": args.resnet_weight,
                    "epoch": args.epochs + stage_epoch,
                    "val_metrics": focus_metrics,
                    "feature_layer": args.feature_layer,
                }, "checkpoints/best_model_kappa.pth")

    best_checkpoint = args.checkpoint_path
    if os.path.exists(best_checkpoint):
        best_state = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion)
    history["test_accuracy"] = test_metrics["accuracy"]
    history["test_resnet_accuracy"] = test_metrics["resnet_accuracy"]
    history["test_efficientnet_accuracy"] = test_metrics["efficientnet_accuracy"]
    history["test_macro_recall"] = test_metrics["macro_recall"]
    history["test_macro_f1"] = test_metrics["macro_f1"]
    history["test_weighted_kappa"] = test_metrics["weighted_kappa"]
    history["test_per_class_recall"] = test_metrics["per_class_recall"]
    history["test_resnet_macro_recall"] = test_metrics["resnet_macro_recall"]
    history["test_efficientnet_macro_recall"] = test_metrics["efficientnet_macro_recall"]
    history["test_resnet_weighted_kappa"] = test_metrics["resnet_weighted_kappa"]
    history["test_efficientnet_weighted_kappa"] = test_metrics["efficientnet_weighted_kappa"]
    history["test_resnet_per_class_recall"] = test_metrics["resnet_per_class_recall"]
    history["test_efficientnet_per_class_recall"] = test_metrics["efficientnet_per_class_recall"]
    history["test_confusion_matrix"] = test_metrics["confusion_matrix"]

    write_comparison_report(test_metrics)

    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot curves
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("Loss"); axes[0].legend()
        axes[1].plot(history["val_accuracy"], label="val_accuracy")
        axes[1].set_title("Val Accuracy"); axes[1].legend()
        axes[2].plot(history["val_macro_recall"], label="val_macro_recall")
        axes[2].set_title("Val Macro Recall"); axes[2].legend()
        plt.tight_layout()
        plt.savefig("outputs/training_curves.png", dpi=150)
        print("Saved outputs/training_curves.png", flush=True)
    except Exception as e:
        print("Could not plot curves:", e, flush=True)

    print(f"\nTraining complete. Best validation {args.selection_metric}: {best_metric:.4f}", flush=True)
    print(f"Test accuracy | ResNet50: {test_metrics['resnet_accuracy']:.4f} | "
          f"EfficientNet-B0: {test_metrics['efficientnet_accuracy']:.4f} | "
          f"Ensemble: {test_metrics['accuracy']:.4f}", flush=True)
    print(f"Test weighted kappa | ResNet50: {test_metrics['resnet_weighted_kappa']:.4f} | "
          f"EfficientNet-B0: {test_metrics['efficientnet_weighted_kappa']:.4f} | "
          f"Ensemble: {test_metrics['weighted_kappa']:.4f}", flush=True)
    print("Severe test confusion row:", test_metrics["confusion_matrix"][3], flush=True)
    print("Proliferate_DR test confusion row:", test_metrics["confusion_matrix"][4], flush=True)
    print(f"Best checkpoint: {args.checkpoint_path}", flush=True)
    if args.save_kappa_checkpoint:
        print(f"Kappa checkpoint: {args.kappa_checkpoint_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, default="data/split")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss focusing parameter")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--feature_layer", type=str, default="layer3", choices=["layer2", "layer3"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_epochs", type=int, default=0, help="Deprecated; final backbone blocks are trainable from epoch 1")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--focus_multiplier", type=float, default=2.5,
                        help="Extra weight applied to Mild, Moderate, and Proliferative classes during training")
    parser.add_argument("--objective", choices=["accuracy", "weighted_ce", "ordinal"], default="ordinal",
                        help="Loss objective: accuracy, inverse-frequency weighted CE, or ordinal CORAL-style loss")
    parser.add_argument("--selection_metric", choices=["accuracy", "macro_recall", "macro_f1"],
                        default="macro_recall", help="Metric used to select the best checkpoint")
    parser.add_argument("--resnet_weight", type=float, default=0.7,
                        help="Ensemble probability weight for ResNet50; EfficientNet gets the remainder")
    parser.add_argument("--oversample_minority", action="store_true",
                        help="Use aggressive augmentation and oversampling for Severe and Proliferate_DR")
    parser.add_argument("--minority_multiplier", type=float, default=2.0,
                        help="Sampling multiplier for Severe and Proliferate_DR")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="Restricted mixup strength for Moderate/Severe/Proliferate_DR pairs")
    parser.add_argument("--second_stage_epochs", type=int, default=3,
                        help="Focus-only fine-tuning epochs after the main stage")
    parser.add_argument("--second_stage_lr_scale", type=float, default=0.1,
                        help="Learning-rate scale for the focus-only fine-tuning stage")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pth",
                        help="Versioned path for the best validation checkpoint")
    parser.add_argument("--kappa_checkpoint_path", type=str, default="checkpoints/best_model_kappa.pth",
                        help="Path for optional weighted-kappa checkpoint")
    parser.add_argument("--save_kappa_checkpoint", action="store_true",
                        help="Also save a weighted-kappa checkpoint")
    parser.add_argument("--recovery_accuracy", action="store_true",
                        help="Use the original plain-CE, standard-augmentation accuracy configuration")
    args = parser.parse_args()

    if args.recovery_accuracy:
        args.objective = "accuracy"
        args.selection_metric = "accuracy"
        args.oversample_minority = False
        args.mixup_alpha = 0.0
        args.second_stage_epochs = 0
        args.save_kappa_checkpoint = False
        print("Recovery accuracy mode: plain cross-entropy, standard augmentation, no oversampling, no mixup, "
              "validation-accuracy checkpoint selection.", flush=True)

    train(args)
