# Model Checkpoints

This folder houses the trained PyTorch `.pth` checkpoint weights for the **RetinaVision AI** ensemble architecture.

> [!NOTE]
> PyTorch model weights (~120 MB each) exceed GitHub's standard 100 MB per-file upload limit. Consequently, binary `.pth` weights are excluded from Git commits via `.gitignore` to prevent repository bloat.

---

## 1. Verified Model Checkpoints

| Checkpoint File | Backbone / Head | Test Accuracy (553 held-out) | Validation QWK | Purpose |
|:---|:---|:---:|:---:|:---|
| `best_model_accuracy_RECOVERED.pth` | Dual-Attention ResNet50 + EfficientNet-B0 (0.7 / 0.3 Fusion) | **86.62%** | **0.8780** | **Primary production checkpoint** used in Flask web application & review report. |
| `best_model_kappa.pth` | Dual-Attention ResNet50 + EfficientNet-B0 | 85.12% | 0.8845 | Optimized specifically for quadratic weighted kappa. |
| `ablation_no_attention.pth` | ResNet50 without Dual Attention Head | 81.34% | 0.8120 | Baseline model for ablation comparison study. |

---

## 2. Obtaining Trained Weights

You can obtain the trained weights in two ways:

### Option A: Train via Google Colab / Kaggle GPU
1. Open the included [`RetinaVision_Colab.ipynb`](../RetinaVision_Colab.ipynb) in Google Colab (or Kaggle Notebooks with T4 GPU accelerator).
2. Run the end-to-end training notebook (typically takes 35–45 minutes for 30 epochs).
3. The trained checkpoint will be saved to `checkpoints/best_model_accuracy_RECOVERED.pth`. Download and place it into this directory.

### Option B: Local GPU Training
Run the training script locally with your GPU:
```bash
python src/train.py --epochs 30 --batch_size 32 --lr 1e-4
```
The checkpoint will be saved automatically to `checkpoints/best_model_accuracy_RECOVERED.pth`.
