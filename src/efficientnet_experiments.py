"""Independent EfficientNet-B0 diagnosis, tuning, and ensemble sweep.

Examples:
    python src/efficientnet_experiments.py --diagnose-checkpoint checkpoints/best_model_accuracy.pth --ensemble-checkpoint checkpoints/best_model_accuracy.pth
  python src/efficientnet_experiments.py --epochs 12 --head-warmup 2 --dropout-values 0.2,0.3,0.4
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append(os.path.dirname(__file__))
from dataset import CLASS_NAMES, get_dataloaders, get_eval_transforms, DRDataset, resolve_split_path
from losses import OrdinalCORALLoss, compute_class_weights
from model import build_ensemble_model


def load_checkpoint(path, device, purpose):
    if not os.path.isfile(path):
        available = [
            candidate for candidate in (
                "checkpoints/best_model_accuracy.pth",
                "checkpoints/best_model_kappa.pth",
            ) if os.path.isfile(candidate)
        ]
        hint = f" Available checkpoints: {', '.join(available)}." if available else " No checkpoint files were found."
        raise FileNotFoundError(
            f"Cannot {purpose}: checkpoint '{path}' does not exist from the current working directory "
            f"'{os.getcwd()}'.{hint} Pass the correct path with the corresponding checkpoint option."
        )
    return torch.load(path, map_location=device)


def build_efficientnet(dropout, pretrained=True, device="cpu"):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    return model.to(device)


def load_efficientnet_branch(checkpoint, device):
    state_dict = checkpoint["model_state_dict"]
    print(f"Checkpoint state_dict keys: {len(state_dict)}")
    print(f"EfficientNet key samples: {[key for key in state_dict if key.startswith(('efficientnet.', 'efficientnet_branch.'))][:5]}")

    prefixes = ("efficientnet.", "efficientnet_branch.")
    branch_state = {
        key[len(prefix):]: value
        for prefix in prefixes
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not branch_state:
        branch_keys = {"features.", "classifier."}
        branch_state = {
            key: value for key, value in state_dict.items()
            if any(key.startswith(prefix) for prefix in branch_keys)
        }
    if not branch_state:
        raise KeyError("No EfficientNet-B0 weights found in checkpoint model_state_dict.")

    print(f"Loading EfficientNet-B0 branch keys: {len(branch_state)}")
    model = build_efficientnet(
        dropout=checkpoint.get("dropout", 0.4), pretrained=False, device=device
    )
    load_result = model.load_state_dict(branch_state, strict=True)
    print(f"EfficientNet load missing keys: {load_result.missing_keys}")
    print(f"EfficientNet load unexpected keys: {load_result.unexpected_keys}")
    model.eval()
    return model


def batch_tensors(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"Expected a (images, labels) batch, received {type(batch).__name__}")
    images, labels = batch
    if not torch.is_tensor(images):
        images = torch.stack(list(images))
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    return images, labels.long()


def set_progressive_stage(model, stage):
    """Stage 0=head, then progressively unfreeze the top MBConv blocks."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    feature_blocks = list(model.features.children())
    blocks_to_open = min(stage, len(feature_blocks))
    for module in feature_blocks[-blocks_to_open:] if blocks_to_open else []:
        for parameter in module.parameters():
            parameter.requires_grad = True


def restricted_mixup(images, labels, alpha):
    eligible = torch.where((labels >= 2) & (labels <= 4))[0]
    if alpha <= 0 or eligible.numel() < 2:
        return images, labels, labels, 1.0
    permutation = torch.arange(labels.numel(), device=labels.device)
    permutation[eligible] = eligible[torch.randperm(eligible.numel(), device=labels.device)]
    lam = float(np.random.beta(alpha, alpha))
    return lam * images + (1 - lam) * images[permutation], labels, labels[permutation], lam


def predictions(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            output = model(images.to(device))
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(output.argmax(1).cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(preds)


def metrics(labels, preds):
    report = classification_report(
        labels, preds, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float((labels == preds).mean()),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "per_class": {name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
        } for name in CLASS_NAMES},
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES)))).tolist(),
    }


def diagnose_existing_efficientnet(args, device):
    checkpoint = load_checkpoint(args.diagnose_checkpoint, device, "diagnose EfficientNet")
    model = load_efficientnet_branch(checkpoint, device)
    dataset = DRDataset(resolve_split_path(args.split_dir, "test"), transform=get_eval_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels, preds = [], []
    with torch.no_grad():
        for batch in loader:
            images, batch_labels = batch_tensors(batch)
            probabilities = model(images.to(device)).softmax(1)
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(probabilities.argmax(1).cpu().numpy().tolist())
    result = metrics(np.asarray(labels), np.asarray(preds))
    print("EfficientNet-B0 baseline precision/recall/F1:")
    for name, values in result["per_class"].items():
        print(f"  {name:16s} precision={values['precision']:.4f} "
              f"recall={values['recall']:.4f} f1={values['f1']:.4f}")
    print("EfficientNet-B0 baseline confusion matrix:")
    print(np.asarray(result["confusion_matrix"]))
    print(f"EfficientNet-B0 baseline test accuracy: {result['accuracy']:.4f}")


def train_one(args, dropout, device, train_loader, val_loader, counts):
    model = build_efficientnet(dropout, pretrained=True, device=device)
    criterion = OrdinalCORALLoss(len(CLASS_NAMES), compute_class_weights(counts))
    best = None
    optimizer = None
    for epoch in range(1, args.epochs + 1):
        stage = 0 if epoch <= args.head_warmup else min(
            len(model.features), 1 + (epoch - args.head_warmup - 1) // args.unfreeze_every
        )
        set_progressive_stage(model, stage)
        if optimizer is None or epoch == args.head_warmup + 1 or (epoch > args.head_warmup + 1 and stage != previous_stage):
            params = [
                {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
            ]
            optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        previous_stage = stage
        model.train()
        for batch_index, batch in enumerate(train_loader):
            images, labels = batch_tensors(batch)
            if batch_index == 0:
                print(f"Training batch: images={type(images).__name__}{tuple(images.shape)}, "
                      f"labels={type(labels).__name__}{tuple(labels.shape)} dtype={labels.dtype}", flush=True)
            images, labels = images.to(device), labels.to(device)
            mixed, first, second, lam = restricted_mixup(images, labels, args.mixup_alpha)
            optimizer.zero_grad()
            output = model(mixed)
            loss = lam * criterion(output, first) + (1 - lam) * criterion(output, second)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_labels, val_preds = predictions(model, val_loader, device)
        val_metrics = metrics(val_labels, val_preds)
        print(f"dropout={dropout:.1f} epoch={epoch:02d} stage={stage} "
              f"val_accuracy={val_metrics['accuracy']:.4f} "
              f"val_macro_recall={val_metrics['macro_recall']:.4f}", flush=True)
        if best is None or val_metrics["macro_recall"] > best["metrics"]["macro_recall"]:
            best = {"metrics": val_metrics, "state_dict": model.state_dict(), "epoch": epoch}
    return model, best


def evaluate_ensemble(args, efficient_state, efficient_dropout, device):
    checkpoint = load_checkpoint(args.ensemble_checkpoint, device, "evaluate the ensemble")
    ensemble = build_ensemble_model(
        num_classes=len(CLASS_NAMES), feature_layer=checkpoint.get("feature_layer", "layer3"),
        pretrained=False, dropout=args.resnet_dropout,
        resnet_weight=args.resnet_weight, device=device
    )
    ensemble.load_state_dict(checkpoint["model_state_dict"])
    efficient = build_efficientnet(efficient_dropout, pretrained=False, device=device)
    efficient.load_state_dict(efficient_state)
    dataset = DRDataset(resolve_split_path(args.split_dir, "test"), transform=get_eval_transforms())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels, branch_outputs = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            resnet_probs, _ = ensemble.branch_probabilities(images)
            efficient_probs = efficient(images).softmax(1)
            labels.extend(batch_labels.numpy().tolist())
            branch_outputs.extend(zip(resnet_probs.cpu().numpy(), efficient_probs.cpu().numpy()))
    labels = np.asarray(labels)
    result = {}
    resnet_predictions = np.asarray([resnet.argmax() for resnet, _ in branch_outputs])
    efficient_predictions = np.asarray([efficient.argmax() for _, efficient in branch_outputs])
    result["resnet50"] = metrics(labels, resnet_predictions)
    result["efficientnet_b0"] = metrics(labels, efficient_predictions)
    print(f"ResNet50 accuracy={result['resnet50']['accuracy']:.4f} | "
          f"EfficientNet-B0 accuracy={result['efficientnet_b0']['accuracy']:.4f}", flush=True)
    for weight in (0.70, 0.75, 0.80):
        predictions_array = np.asarray([
            weight * resnet + (1 - weight) * efficient
            for resnet, efficient in branch_outputs
        ]).argmax(1)
        result[str(weight)] = metrics(labels, predictions_array)
        print(f"ensemble resnet_weight={weight:.2f} accuracy={result[str(weight)]['accuracy']:.4f} "
              f"macro_recall={result[str(weight)]['macro_recall']:.4f}", flush=True)
    return result


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.diagnose_checkpoint:
        diagnose_existing_efficientnet(args, device)
    train_loader, val_loader, _, train_ds, _, _ = get_dataloaders(
        split_dir=args.split_dir, batch_size=args.batch_size, num_workers=args.num_workers,
        oversample_minority=True, minority_multiplier=args.minority_multiplier
    )
    print("EfficientNet input normalization: mean=(0.485, 0.456, 0.406), "
          "std=(0.229, 0.224, 0.225)")
    print("EfficientNet schedule: classifier warmup, then top MBConv blocks progressively; "
          f"base_lr={args.lr:.2e}, head_warmup={args.head_warmup}")
    counts = train_ds.class_counts()
    all_results = {}
    best_model = None
    best_dropout = None
    for dropout in args.dropout_values:
        model, best = train_one(args, dropout, device, train_loader, val_loader, counts)
        all_results[str(dropout)] = best["metrics"]
        if best_model is None or best["metrics"]["macro_recall"] > all_results[str(best_dropout)]["macro_recall"]:
            best_model, best_dropout = best, dropout
    os.makedirs(args.output_dir, exist_ok=True)
    efficient_path = os.path.join(args.output_dir, "best_efficientnet_b0.pth")
    torch.save({"model_state_dict": best_model["state_dict"], "dropout": best_dropout}, efficient_path)
    print(f"Saved independent EfficientNet checkpoint: {efficient_path}")
    all_results["ensemble_weight_sweep"] = evaluate_ensemble(
        args, best_model["state_dict"], best_dropout, device
    )
    report_path = os.path.join(args.output_dir, "efficientnet_experiments.json")
    with open(report_path, "w") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", default="data/split")
    parser.add_argument("--ensemble-checkpoint", default="checkpoints/best_model_kappa.pth")
    parser.add_argument("--diagnose-checkpoint", default=None,
                        help="Existing ensemble checkpoint used to print EfficientNet-only test metrics")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--head-warmup", type=int, default=2)
    parser.add_argument("--unfreeze-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-values", type=lambda value: [float(x) for x in value.split(",")], default=[0.2, 0.3, 0.4])
    parser.add_argument("--minority-multiplier", type=float, default=2.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--resnet-dropout", type=float, default=0.4)
    parser.add_argument("--resnet-weight", type=float, default=0.7)
    parser.add_argument("--output-dir", default="outputs/efficientnet")
    args = parser.parse_args()
    main(args)