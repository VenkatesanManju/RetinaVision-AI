# Dataset Setup & Guidelines

This repository includes 5 representative sample retina fundus images under `data/samples/` (one for each diabetic retinopathy severity grade) to allow immediate testing and demonstration without downloading the full dataset.

---

## 1. Dataset Overview

* **Source**: [Kaggle APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
* **Preprocessed Distribution**: [Sovit Rath's Gaussian Filtered 224x224 Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)
* **Image Count**: 3,661 preprocessed fundus photographs
* **Image Resolution**: 224 × 224 pixels
* **Illumination Normalization**: Gaussian blur subtraction (Ben Graham's method) to remove camera-specific illumination variance and enhance micro-lesions (microaneurysms, hemorrhages, and exudates).

---

## 2. Five Clinical Severity Classes

| Class ID | Stage Name | Description | Total Images |
|:---:|:---|:---|:---:|
| **0** | **No DR** | Healthy retina, no visible diabetic lesions | 1,805 |
| **1** | **Mild DR** | Early microaneurysms only | 370 |
| **2** | **Moderate DR** | Microaneurysms, hemorrhages, hard/soft exudates | 999 |
| **3** | **Severe DR** | >20 intraretinal hemorrhages in 4 quadrants, venous beading, IRMA | 193 |
| **4** | **Proliferative DR** | Neovascularization (abnormal new vessels), vitreous/preretinal hemorrhage | 294 |

---

## 3. Stratified Split (70% Train / 15% Val / 15% Held-Out Test)

The dataset was partitioned using stratified sampling to preserve the clinical class imbalance across all sets:

| Class | Train (70%) | Validation (15%) | Test (15%) | Total |
|:---|:---:|:---:|:---:|:---:|
| **No DR** | 1,263 | 270 | 272 | 1,805 |
| **Mild** | 259 | 55 | 56 | 370 |
| **Moderate** | 699 | 149 | 151 | 999 |
| **Severe** | 135 | 28 | 30 | 193 |
| **Proliferative DR** | 206 | 44 | 44 | 294 |
| **Total** | **2,562** | **546** | **553** | **3,661** |

---

## 4. How to Download and Split Full Dataset

### Option A: Using Kaggle CLI
```bash
# 1. Download preprocessed dataset
kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-gaussian-filtered

# 2. Extract into data/raw/
unzip diabetic-retinopathy-224x224-gaussian-filtered.zip -d data/raw/

# 3. Generate stratified splits
python src/prepare_data.py --raw_dir data/raw/gaussian_filtered_images/gaussian_filtered_images --out_dir data/split
```

### Option B: Cloud GPU Training (Kaggle Notebook / Google Colab)
If you are training on Kaggle Notebooks, simply add `sovitrath/diabetic-retinopathy-224x224-gaussian-filtered` as an input dataset. It mounts automatically at `/kaggle/input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/`.
