"""Unified multi-task model
"""

import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        import gdown

        # Download weights (you will fill actual IDs later)
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)

        # Initialize models
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # Load weights
        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        self.localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model."""

        cls_out = self.classifier(x)     # [B, num_breeds]
        loc_out = self.localizer(x)      # [B, 4]
        seg_out = self.segmenter(x)      # [B, seg_classes, H, W]

        return {
            "classification": cls_out,
            "localization": loc_out,
            "segmentation": seg_out,
        }