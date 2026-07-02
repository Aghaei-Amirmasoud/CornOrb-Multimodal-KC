"""
models/fusion_net.py
MultimodalFusionNet: ResNet-18 image branch + ClinicalMLP branch + fusion head.

Architecture:
    4 Corneal Maps (12ch)             Clinical Features (7d)
          │                                    │
    [ResNet-18]                     [MLP: 7→64→128]
    (12ch input, pretrained          ReLU + Dropout
     RGB weights averaged)                     │
    → 512-d feature vector            128-d feature vector
          │                                    │
          └──────────── Concat ────────────────┘
                           │
                [Fusion MLP: 640→256→64→2]
                           │
                      CrossEntropyLoss
"""

import torch
import torch.nn as nn
from torchvision import models

from models.clinical_mlp import ClinicalMLP


class MultimodalFusionNet(nn.Module):
    """
    Late-fusion multimodal network for binary keratoconus classification.

    Args:
        n_clinical_features : number of clinical input features
        n_classes           : number of output classes (default 2)
        mode                : "fused" | "image_only" | "clinical_only"
    """

    def __init__(self, n_clinical_features: int, n_classes: int = 2,
                 mode: str = "fused"):
        super().__init__()
        self.mode = mode

        # ── Image branch: ResNet-18 pretrained on ImageNet ────────────────────
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv from 3 → 12 channels (average pretrained RGB weights)
        old_w = backbone.conv1.weight.data          # (64, 3, 7, 7)
        new_w = old_w.repeat(1, 4, 1, 1) / 4.0     # (64, 12, 7, 7)
        backbone.conv1 = nn.Conv2d(
            12, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.conv1.weight.data = new_w

        # Remove final FC — keep 512-d feature vector
        self.image_branch = nn.Sequential(*list(backbone.children())[:-1])
        self.img_feat_dim  = 512

        # ── Clinical branch ───────────────────────────────────────────────────
        self.clin_feat_dim    = 128
        self.clinical_branch  = ClinicalMLP(n_clinical_features,
                                            hidden=self.clin_feat_dim)

        # ── Fusion head ───────────────────────────────────────────────────────
        if mode == "fused":
            fusion_in = self.img_feat_dim + self.clin_feat_dim   # 640
        elif mode == "image_only":
            fusion_in = self.img_feat_dim                         # 512
        elif mode == "clinical_only":
            fusion_in = self.clin_feat_dim                        # 128
        else:
            raise ValueError(f"Unknown mode: {mode!r}. "
                             "Use 'fused', 'image_only', or 'clinical_only'.")

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, images, clinical):
        if self.mode in ("fused", "image_only"):
            img_feat = self.image_branch(images).flatten(1)   # (B, 512)
        if self.mode in ("fused", "clinical_only"):
            clin_feat = self.clinical_branch(clinical)         # (B, 128)

        if self.mode == "fused":
            x = torch.cat([img_feat, clin_feat], dim=1)
        elif self.mode == "image_only":
            x = img_feat
        elif self.mode == "clinical_only":
            x = clin_feat

        return self.fusion_head(x)
