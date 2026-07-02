"""
models/clinical_mlp.py
Lightweight MLP branch for structured clinical features.
"""

import torch.nn as nn


class ClinicalMLP(nn.Module):
    """
    Two-layer MLP with BatchNorm and Dropout for tabular clinical features.

    Args:
        in_features : number of input clinical features (e.g. 7)
        hidden      : size of the output embedding (default 128)
    """

    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)
