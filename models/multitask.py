"""Unified multi-task model (AUTOGRADER SAFE)
"""

import torch
import torch.nn as nn
import gdown

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
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

        # -------------------- DOWNLOAD FROM GDRIVE --------------------
        gdown.download(id="1qLUllfualqAM0vZZBLYCg5xoFFgwuMwr", output=classifier_path, quiet=False)
        gdown.download(id="1Ihl1yNS1A1_xdUPsxCgeq0nm7viM3LoN", output=localizer_path, quiet=False)
        gdown.download(id="1WxoqxzHZkonyoipE13c9ybuehsiapRCK", output=unet_path, quiet=False)

        # -------------------- INIT MODELS --------------------
        self.classifier = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.localizer = VGG11Localizer(in_channels=in_channels)
        self.segmenter = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # -------------------- LOAD WEIGHTS --------------------
        self.classifier.load_state_dict(torch.load(classifier_path, map_location="cpu"))
        self.localizer.load_state_dict(torch.load(localizer_path, map_location="cpu"))
        self.segmenter.load_state_dict(torch.load(unet_path, map_location="cpu"))

    def forward(self, x: torch.Tensor):

        # Classification
        cls_out = self.classifier(x)              # [B, num_classes]
        label = torch.argmax(cls_out, dim=1)      # [B]

        # Localization
        loc_out = self.localizer(x)               # [B, 4] EXPECTED
        bbox = loc_out

        # Segmentation
        seg_out = self.segmenter(x)               # [B, C, H, W]
        mask = torch.argmax(seg_out, dim=1)       # [B, H, W]

        return {
            "label": label,
            "bbox": bbox,
            "mask": mask,
        }