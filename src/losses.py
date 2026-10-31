"""
losses.py
---------
Class-imbalance-aware loss functions.

We provide:
    1. compute_class_weights() - inverse-frequency weights for CrossEntropyLoss
    2. FocalLoss - multi-class focal loss with optional per-class alpha weights,
       which down-weights easy (usually majority-class) examples and focuses
       gradient on hard/minority examples (Mild, Severe, Proliferative here).

Default training uses FocalLoss(alpha=class_weights, gamma=2.0), which in
practice gives a bigger boost to Mild/Moderate recall than plain weighted CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_class_weights(class_counts, mode="inverse", focus_indices=None, focus_multiplier=1.0):
    """
    class_counts: array-like of per-class sample counts (train split only).
    Returns a torch.FloatTensor of per-class weights, normalized so mean == 1.

    focus_indices: optional indices to emphasize more strongly (e.g., Mild, Moderate,
                   Proliferative). focus_multiplier increases their weight above the
                   base inverse-frequency weighting.
    """
    counts = np.array(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 1)  # avoid div-by-zero

    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError("mode must be 'inverse' or 'inverse_sqrt'")

    if focus_indices is not None:
        focus_indices = list(focus_indices)
        for idx in focus_indices:
            if 0 <= idx < len(weights):
                weights[idx] *= focus_multiplier

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    L = -alpha_c * (1 - p_c)^gamma * log(p_c)

    alpha: None, scalar, or per-class tensor (recommended: class_weights from
           compute_class_weights).
    gamma: focusing parameter; higher gamma -> more focus on hard examples.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        pt = probs.gather(1, targets).squeeze(1)

        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            at = alpha.gather(0, targets.squeeze(1))
            loss = loss * at

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalCORALLoss(nn.Module):
    """Ordinal loss for five ordered DR grades."""

    def __init__(self, num_classes=5, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        probabilities = F.softmax(logits, dim=1)
        cumulative = torch.cumsum(probabilities, dim=1)
        greater_probability = (1.0 - cumulative[:, :-1]).clamp(1e-6, 1.0 - 1e-6)
        cumulative_logits = torch.logit(greater_probability)
        thresholds = torch.arange(self.num_classes - 1, device=targets.device).unsqueeze(0)
        ordinal_targets = (targets.unsqueeze(1) > thresholds).float()
        loss = F.binary_cross_entropy_with_logits(
            cumulative_logits, ordinal_targets, reduction="none"
        ).mean(dim=1)
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(logits.device).gather(0, targets)
        return loss.mean()
