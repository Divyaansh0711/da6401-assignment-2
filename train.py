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


# ---------------------- CLASSIFIER ----------------------
def train_classifier():
    dropout_p = 0.2
    use_batchnorm = False

    wandb.init(
        project="da6401",
        name="classifier_bn_false",
        config={
            "task": "classification",
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "dropout": dropout_p
        }
    )

    # model = VGG11Classifier().to(DEVICE)
    
    model = VGG11Classifier(dropout_p=dropout_p,use_batchnorm=use_batchnorm).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if i % 50 == 0:
                print(f"[Classifier][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_acc = correct / total
        val_acc = val_correct / val_total

        print(f"[Classifier] Epoch {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/accuracy": train_acc,
            "val/accuracy": val_acc
        })

    torch.save(model.state_dict(), f"classifier_bn_{use_batchnorm}.pth")
    wandb.finish()


# ---------------------- LOCALIZER ----------------------
def train_localizer():
    wandb.init(
        project="da6401",
        name="localizer_base",
        config={
            "task": "localization",
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "loss": "MSE + IoU"
        }
    )

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
            "train/loss": train_loss,
            "val/loss": val_loss
        })

    torch.save(model.state_dict(), "localizer.pth")
    wandb.finish()


# ---------------------- SEGMENTER ----------------------
def train_segmenter():
    transfer_mode = "strict"   # choose from: "strict", "partial", "full"

    wandb.init(
        project="da6401",
        name=f"segmenter_{transfer_mode}",
        config={
            "task": "segmentation",
            "transfer_mode": transfer_mode,
            "lr": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "loss": "CrossEntropy"
        }
    )

    model = VGG11UNet().to(DEVICE)

    # -------------------- Transfer learning setup --------------------
    if transfer_mode == "strict":
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif transfer_mode == "partial":
        # Freeze early blocks, fine-tune later ones
        for param in model.encoder.block1.parameters():
            param.requires_grad = False
        for param in model.encoder.block2.parameters():
            param.requires_grad = False
        for param in model.encoder.block3.parameters():
            param.requires_grad = False

        for param in model.encoder.block4.parameters():
            param.requires_grad = True
        for param in model.encoder.block5.parameters():
            param.requires_grad = True

    elif transfer_mode == "full":
        for param in model.encoder.parameters():
            param.requires_grad = True

    else:
        raise ValueError("transfer_mode must be one of: 'strict', 'partial', 'full'")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

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
                print(f"[Seg][{transfer_mode}][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

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

        print(f"[Segmentation][{transfer_mode}] Epoch {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss
        })

    torch.save(model.state_dict(), f"unet_{transfer_mode}.pth")
    wandb.finish()


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    train_segmenter()  # run ONE at a time