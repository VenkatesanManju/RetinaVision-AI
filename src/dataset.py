"""
dataset.py
----------
PyTorch Dataset / DataLoader helpers for the split DR dataset, plus
fundus-appropriate data augmentation.

Augmentation choices (deliberately conservative on color):
    - Random rotation (retina images have no canonical "up") 
    - Random horizontal & vertical flip
    - Slight brightness/contrast jitter only (no hue/saturation shifts,
      since color shifts can wash out or exaggerate hemorrhages/exudates
      that the model needs to key on)
    - Random slight zoom/crop to simulate camera distance variation
    - Normalization with ImageNet stats (since backbone is ImageNet-pretrained)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(aggressive=False):
    if aggressive:
        rotation = 35
        crop_scale = (0.82, 1.0)
        jitter = 0.35
        flip_probability = 0.6
    else:
        rotation = 25
        crop_scale = (0.9, 1.0)
        jitter = 0.15
        flip_probability = 0.5

    transform_ops = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=flip_probability),
        transforms.RandomVerticalFlip(p=flip_probability),
        transforms.RandomRotation(degrees=rotation),
        transforms.RandomResizedCrop(IMG_SIZE, scale=crop_scale, ratio=(0.95, 1.05)),
        transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=0.2, hue=0.03),
    ]
    if aggressive:
        transform_ops.extend([
            transforms.RandomApply([
                transforms.ElasticTransform(alpha=25.0, sigma=4.0),
            ], p=0.35),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.14), ratio=(0.3, 3.3)),
        ])
    else:
        transform_ops.append(transforms.ToTensor())
    transform_ops.extend([
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(transform_ops)


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DRDataset(Dataset):
    def __init__(self, root_dir, transform=None, minority_transform=None):
        """
        root_dir: e.g. data/split/train or Train (must contain one subfolder per class)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.minority_transform = minority_transform
        self.samples = []  # list of (path, label)

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                # Fallback to case-insensitive match
                if os.path.exists(root_dir):
                    for item in os.listdir(root_dir):
                        if item.lower() == cls_name.lower() and os.path.isdir(os.path.join(root_dir, item)):
                            cls_dir = os.path.join(root_dir, item)
                            break
            if not os.path.isdir(cls_dir):
                continue

            label = CLASS_NAMES.index(cls_name)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Check dataset folder structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.minority_transform and label in (3, 4):
            image = self.minority_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self):
        counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for _, label in self.samples:
            counts[label] += 1
        return counts


def resolve_split_path(split_dir, split_name):
    candidates = [
        os.path.join(split_dir, split_name),
        os.path.join(split_dir, split_name.capitalize()),
        os.path.join(".", split_name),
        os.path.join(".", split_name.capitalize()),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(split_dir, split_name)


def get_dataloaders(split_dir="data/split", batch_size=32, num_workers=0,
                    oversample_minority=False, minority_multiplier=2.0):
    train_path = resolve_split_path(split_dir, "train")
    val_path = resolve_split_path(split_dir, "val")
    test_path = resolve_split_path(split_dir, "test")

    train_transform = get_train_transforms()
    minority_transform = get_train_transforms(aggressive=True) if oversample_minority else None
    train_ds = DRDataset(train_path, transform=train_transform, minority_transform=minority_transform)
    val_ds = DRDataset(val_path, transform=get_eval_transforms())
    test_ds = DRDataset(test_path, transform=get_eval_transforms())

    pin_memory = torch.cuda.is_available()
    if oversample_minority:
        sample_weights = torch.ones(len(train_ds), dtype=torch.double)
        for index, (_, label) in enumerate(train_ds.samples):
            if label in (3, 4):
                sample_weights[index] = minority_multiplier
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders()
    print("Train class counts:", dict(zip(CLASS_NAMES, train_ds.class_counts())))
    print("Val class counts:  ", dict(zip(CLASS_NAMES, val_ds.class_counts())))
    print("Test class counts: ", dict(zip(CLASS_NAMES, test_ds.class_counts())))
