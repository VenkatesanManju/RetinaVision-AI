"""
fast_train.py
-------------
Feature-cached fast training for Dual Attention ResNet50 DR Classifier.
Pre-extracts backbone features once (~2 mins) and trains the Dual Attention
unit + classification head (~10s for 35 epochs), followed by fine-tuning.
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders, CLASS_NAMES
from model import build_model
from losses import compute_class_weights, FocalLoss
from evaluate import run_evaluation


def extract_features(model, loader, device):
    model.eval()
    all_feats = []
    all_labels = []
    print("   Extracting backbone features...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model._forward_backbone(images)  # B x 1024 x 14 x 14
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"   Extracted {len(all_feats)} samples in {time.time()-t0:.1f}s. Shape: {all_feats.shape}", flush=True)
    return all_feats, all_labels


def train_head_on_features(model, train_feats, train_labels, val_feats, val_labels,
                           criterion, epochs=35, batch_size=32, lr=1e-3, device="cpu"):
    train_ds = TensorDataset(train_feats, train_labels)
    val_ds = TensorDataset(val_feats, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    head_params = list(model.dual_attention.parameters()) + list(model.head_conv.parameters()) + \
                  list(model.head_bn.parameters()) + list(model.fc1.parameters()) + list(model.fc2.parameters())
    optimizer = AdamW(head_params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print("\n--- Phase 1: Feature-Cached Attention & Classifier Head Training ---", flush=True)
    best_macro_recall = -1.0
    best_head_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()

            attended = model.dual_attention(feats)
            h = model.head_conv(attended)
            h = model.head_bn(h)
            h = model.head_relu(h)
            pooled = model.gap(h).flatten(1)
            pooled = torch.relu(model.fc1(pooled))
            pooled = model.dropout(pooled)
            logits = model.fc2(pooled)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluate on validation features
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                attended = model.dual_attention(feats)
                h = model.head_conv(attended)
                h = model.head_bn(h)
                h = model.head_relu(h)
                pooled = model.gap(h).flatten(1)
                pooled = torch.relu(model.fc1(pooled))
                logits = model.fc2(pooled)

                loss = criterion(logits, labels)
                val_loss += loss.item()
                n_val_batches += 1

                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        macro_recall = float(recall_score(val_targets, val_preds, average="macro", zero_division=0))
        accuracy = float((val_preds == val_targets).mean())
        scheduler.step(macro_recall)

        elapsed = time.time() - t0
        print(f"Phase 1 | Epoch {epoch:02d}/{epochs:02d} | train_loss={running_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss/n_val_batches:.4f} | val_acc={accuracy:.4f} | "
              f"val_macro_recall={macro_recall:.4f} | {elapsed:.2f}s", flush=True)

        if macro_recall > best_macro_recall:
            best_macro_recall = macro_recall
            best_head_state = {
                "dual_attention": model.dual_attention.state_dict(),
                "head_conv": model.head_conv.state_dict(),
                "head_bn": model.head_bn.state_dict(),
                "fc1": model.fc1.state_dict(),
                "fc2": model.fc2.state_dict(),
            }

    if best_head_state is not None:
        model.dual_attention.load_state_dict(best_head_state["dual_attention"])
        model.head_conv.load_state_dict(best_head_state["head_conv"])
        model.head_bn.load_state_dict(best_head_state["head_bn"])
        model.fc1.load_state_dict(best_head_state["fc1"])
        model.fc2.load_state_dict(best_head_state["fc2"])

    print(f"Phase 1 complete. Best val macro recall: {best_macro_recall:.4f}", flush=True)
    return best_macro_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", type=str, default="data/split")
    parser.add_argument("--phase1_epochs", type=int, default=35)
    parser.add_argument("--fine_tune_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(
        split_dir=args.split_dir, batch_size=args.batch_size, num_workers=0
    )

    print("Train class counts:", dict(zip(CLASS_NAMES, train_ds.class_counts())), flush=True)
    class_weights = compute_class_weights(train_ds.class_counts(), mode="inverse")
    print("Class weights (alpha):", class_weights.tolist(), flush=True)

    criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
    model = build_model(num_classes=5, feature_layer="layer3", pretrained=True, dropout=0.4, device=device)

    # Step 1: Pre-extract backbone features for Train and Val
    train_feats, train_labels = extract_features(model, train_loader, device)
    val_feats, val_labels = extract_features(model, val_loader, device)

    # Step 2: Train Attention + Head on extracted features (super fast!)
    best_macro_recall = train_head_on_features(
        model, train_feats, train_labels, val_feats, val_labels,
        criterion, epochs=args.phase1_epochs, batch_size=args.batch_size, lr=args.lr, device=device
    )

    # Step 3: End-to-end fine-tuning for 2 epochs
    if args.fine_tune_epochs > 0:
        print("\n--- Phase 2: End-to-End Fine-Tuning ---", flush=True)
        for p in model.layer3.parameters():
            p.requires_grad = True

        optimizer = AdamW([
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": filter(lambda p: p.requires_grad and p not in set(model.layer3.parameters()), model.parameters()), "lr": 1e-4},
        ], weight_decay=1e-4)

        for ft_epoch in range(1, args.fine_tune_epochs + 1):
            model.train()
            running_loss = 0.0
            t0 = time.time()
            for step, (images, labels) in enumerate(train_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item()
                if step % 20 == 0 or step == len(train_loader):
                    print(f"   FT Batch [{step:02d}/{len(train_loader):02d}] loss={loss.item():.4f}", flush=True)

            # Evaluate full model on validation set
            model.eval()
            val_preds, val_labels_list = [], []
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    n_val += 1
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_labels_list.extend(labels.cpu().numpy())

            val_preds = np.array(val_preds)
            val_labels_arr = np.array(val_labels_list)
            macro_recall = float(recall_score(val_labels_arr, val_preds, average="macro", zero_division=0))
            accuracy = float((val_preds == val_labels_arr).mean())

            elapsed = time.time() - t0
            print(f"Phase 2 FT Epoch {ft_epoch}/{args.fine_tune_epochs} | train_loss={running_loss/len(train_loader):.4f} | "
                  f"val_loss={val_loss/n_val:.4f} | val_acc={accuracy:.4f} | val_macro_recall={macro_recall:.4f} | {elapsed:.1f}s", flush=True)

            if macro_recall > best_macro_recall:
                best_macro_recall = macro_recall

    # Save final model checkpoint
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "epoch": args.phase1_epochs + args.fine_tune_epochs,
        "val_metrics": {"macro_recall": best_macro_recall},
        "feature_layer": "layer3",
    }
    torch.save(checkpoint_dict, "checkpoints/best_model.pth")
    print(f"\nModel saved successfully to checkpoints/best_model.pth (val_macro_recall={best_macro_recall:.4f})", flush=True)

    # Run evaluation script on held-out test split
    print("\n--- Running Evaluation on Held-Out Test Set ---", flush=True)
    eval_args = argparse.Namespace(checkpoint="checkpoints/best_model.pth", split_dir=args.split_dir, batch_size=32)
    run_evaluation(eval_args)


if __name__ == "__main__":
    main()
