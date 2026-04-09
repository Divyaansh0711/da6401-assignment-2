import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordPetsDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4


def get_loaders():
    train_dataset = OxfordPetsDataset(split="train")
    val_dataset = OxfordPetsDataset(split="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def train_classifier():
    wandb.init(project="da6401", name="classifier")

    model = VGG11Classifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 50 == 0:
                print(f"[Classifier][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        acc = correct / total

        print(f"[Classifier] Epoch {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Acc: {acc:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": acc
        })

    torch.save(model.state_dict(), "classifier.pth")
    wandb.finish()


def train_localizer():
    wandb.init(project="da6401", name="localizer")

    model = VGG11Localizer().to(DEVICE)
    criterion_mse = nn.MSELoss()
    criterion_iou = IoULoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            boxes = batch["bbox"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion_mse(preds, boxes) + criterion_iou(preds, boxes)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 50 == 0:
                print(f"[Localizer][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                boxes = batch["bbox"].to(DEVICE)

                preds = model(imgs)
                loss = criterion_mse(preds, boxes) + criterion_iou(preds, boxes)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"[Localizer] Epoch {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    torch.save(model.state_dict(), "localizer.pth")
    wandb.finish()


def train_segmenter():
    wandb.init(project="da6401", name="segmenter")

    model = VGG11UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 50 == 0:
                print(f"[Seg][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                preds = model(imgs)
                loss = criterion(preds, masks)

                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"[Segmentation] Epoch {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    torch.save(model.state_dict(), "unet.pth")
    wandb.finish()


if __name__ == "__main__":
    train_classifier()
    train_localizer()
    train_segmenter()