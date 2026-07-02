"""
data/dataset.py
CornOrbDataset class and DataLoader factory.
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from config import IMG_SIZE, BATCH_SIZE, MAP_TYPES, IMAGE_ROOT


# ── Transforms ────────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class CornOrbDataset(Dataset):
    """
    PyTorch Dataset for the CornOrb multimodal keratoconus dataset.

    Returns per eye:
        stacked_maps : (4*3, H, W) tensor  — 4 maps loaded as RGB (3ch each)
        clinical     : (n_features,) tensor — standardized clinical features
        label        : int                  — 0=Normal, 1=Keratoconus

    Image path convention:
        {image_root}/{patient_code}/{eye}/{patient_code}_{eye}_{MapType}.png
        e.g. ORBSCAN_Dataset/125GC/OD/125GC_OD_Anterior.png
    """

    def __init__(self, dataframe, clinical_cols, transform=None,
                 image_root: str = IMAGE_ROOT):
        self.df            = dataframe.reset_index(drop=True)
        self.clinical_cols = clinical_cols
        self.transform     = transform
        self.image_root    = image_root

    def __len__(self):
        return len(self.df)

    def _load_map(self, patient_code: str, eye: str, map_type: str) -> Image.Image:
        path = os.path.join(
            self.image_root,
            str(patient_code),
            eye,
            f"{patient_code}_{eye}_{map_type}.png",
        )
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        # Fallback: black image if file missing
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

    def __getitem__(self, idx):
        row          = self.df.iloc[idx]
        patient_code = row["patient_code"]
        eye          = row["eye"]           # "OD" or "OS"
        label        = int(row["label"])

        # Load & transform all 4 corneal maps → concatenate to (12, H, W)
        map_tensors = []
        for m in MAP_TYPES:
            img = self._load_map(patient_code, eye, m)
            if self.transform:
                img = self.transform(img)
            map_tensors.append(img)
        stacked = torch.cat(map_tensors, dim=0)

        # Clinical features
        clinical = torch.tensor(
            row[self.clinical_cols].values.astype(np.float32)
        )

        return stacked, clinical, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(train_df, val_df, test_df, clinical_cols,
                      batch_size: int = BATCH_SIZE, num_workers: int = 2):
    """
    Build train / val / test DataLoaders.
    Train loader uses WeightedRandomSampler to handle class imbalance.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = CornOrbDataset(train_df, clinical_cols, transform=train_transform)
    val_ds   = CornOrbDataset(val_df,   clinical_cols, transform=val_transform)
    test_ds  = CornOrbDataset(test_df,  clinical_cols, transform=val_transform)

    # Weighted sampler — oversamples minority class during training
    labels       = train_df["label"].values
    class_counts = np.bincount(labels)
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights),
                                         replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
