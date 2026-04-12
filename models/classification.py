"""Classification components
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier=VGG11Encoder+ClassificationHead."""

    def __init__(
        self,
        num_classes:int=37,
        in_channels:int=3,
        dropout_p:float=0.5,
        use_batchnorm: bool=True,
    ):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
            use_batchnorm: Whether to use BatchNorm in the encoder.
        """
        super().__init__()

        self.encoder=VGG11Encoder(
            in_channels=in_channels,
            use_batchnorm=use_batchnorm
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),

            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model."""
        x=self.encoder(x)        # [B, 512, 7, 7]
        x=self.classifier(x)     # [B, num_classes]
        return x