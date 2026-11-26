"""
models.py — small reusable networks.

``SmallAudioCNN`` is the default model of the typed task bases: a compact,
shape-agnostic conv net over ``(batch, 1, F, T)`` feature maps (any feature
size, any segment length). A good baseline; swap in your own ``nn.Module`` by
overriding ``build_model``.
"""

from __future__ import annotations

import torch.nn as nn


class SmallAudioCNN(nn.Module):
    """Three conv blocks → global pooling → MLP head with ``out_dim`` outputs."""

    def __init__(self, out_dim: int, channels: int = 32, dropout: float = 0.3):
        super().__init__()
        c = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, 3, padding=1), nn.BatchNorm2d(4 * c), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(4 * c * 16, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dim),
        )

    def forward(self, x):
        if x.dim() == 2:          # raw waveform (B, T) → (B, 1, 1, T)
            x = x[:, None, None, :]
        elif x.dim() == 3:        # (B, F, T) → (B, 1, F, T)
            x = x.unsqueeze(1)
        return self.head(self.features(x))
