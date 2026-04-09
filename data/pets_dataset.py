"""Dataset for Oxford-IIIT Pet.
"""

import os
import torch
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset


class OxfordPetsDataset(Dataset):
    def __init__(self, root="data", split="train"):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.trimap_dir = os.path.join(root, "annotations/trimaps")
        self.xml_dir = os.path.join(root, "annotations/xmls")

        split_file = "trainval.txt" if split == "train" else "test.txt"
        split_path = os.path.join(root, "annotations", split_file)

        self.samples = []
        with open(split_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                name = parts[0]
                label = int(parts[1]) - 1
                self.samples.append((name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]

        # ---- Image ----
        img_path = os.path.join(self.image_dir, name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image = image.resize((224, 224))
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # ---- Mask ----
        mask_path = os.path.join(self.trimap_dir, name + ".png")
        mask = Image.open(mask_path)
        mask = mask.resize((224, 224))
        mask = torch.tensor(np.array(mask)).long()

        # FIX: make labels 0-based
        mask = mask - 1

        # ---- Bounding box ----
        xml_path = os.path.join(self.xml_dir, name + ".xml")

        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root_xml = tree.getroot()

            obj = root_xml.find("object")
            bndbox = obj.find("bndbox")

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            x_scale = 224 / orig_w
            y_scale = 224 / orig_h

            bbox = torch.tensor([
                x_center * x_scale,
                y_center * y_scale,
                width * x_scale,
                height * y_scale,
            ], dtype=torch.float32)
        else:
            # fallback: full image box
            bbox = torch.tensor([112.0, 112.0, 224.0, 224.0], dtype=torch.float32)
        

        return {
            "image": image,
            "label": torch.tensor(label),
            "bbox": bbox,
            "mask": mask,
        }