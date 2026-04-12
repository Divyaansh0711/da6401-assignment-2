"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer with bbox + confidence prediction."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.backbone_head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
        )

        self.bbox_head = nn.Linear(4096, 4)   # [x_center, y_center, w, h]
        # self.conf_head = nn.Linear(4096, 1)   # confidence logit

    def forward(self, x: torch.Tensor):
        """Forward pass for localization model."""
        x = self.encoder(x)              # [B, 512, 7, 7]
        x = self.backbone_head(x)        # [B, 4096]

        bbox = torch.sigmoid(self.bbox_head(x)) * 224.0         # [B, 4]
        # confidence = torch.sigmoid(self.conf_head(x))  # [B, 1] in [0,1]

        return bbox