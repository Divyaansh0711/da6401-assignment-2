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
EPOCHS = 20
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
    use_batchnorm = True

    wandb.init(
        project="da6401",
        name="classifier_imporoved",
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

    torch.save(model.state_dict(), f"classifier_improved.pth")
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
            "loss": "MSE + IoU + BCE(confidence)"
        }
    )

    model = VGG11Localizer().to(DEVICE)
    criterion_mse = nn.MSELoss()
    criterion_iou = IoULoss()
    criterion_conf = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loader, val_loader = get_loaders()

    def compute_iou(pred_boxes, target_boxes, eps=1e-6):
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)

        union = pred_area + target_area - inter_area + eps
        iou = inter_area / union

        return iou.mean().item()

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_iou = 0
        train_conf = 0

        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            boxes = batch["bbox"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs)
            pred_boxes = outputs["bbox"]
            pred_conf = outputs["confidence"]

            target_conf = torch.ones_like(pred_conf)

            loss_bbox = criterion_mse(pred_boxes, boxes) + criterion_iou(pred_boxes, boxes)
            loss_conf = criterion_conf(pred_conf, target_conf)
            loss = loss_bbox + loss_conf

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += compute_iou(pred_boxes.detach(), boxes)
            train_conf += pred_conf.mean().item()

            if i % 50 == 0:
                print(f"[Localizer][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        val_iou = 0
        val_conf = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                boxes = batch["bbox"].to(DEVICE)

                outputs = model(imgs)
                pred_boxes = outputs["bbox"]
                pred_conf = outputs["confidence"]

                target_conf = torch.ones_like(pred_conf)

                loss_bbox = criterion_mse(pred_boxes, boxes) + criterion_iou(pred_boxes, boxes)
                loss_conf = criterion_conf(pred_conf, target_conf)
                loss = loss_bbox + loss_conf

                val_loss += loss.item()
                val_iou += compute_iou(pred_boxes, boxes)
                val_conf += pred_conf.mean().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_iou /= len(train_loader)
        val_iou /= len(val_loader)
        train_conf /= len(train_loader)
        val_conf /= len(val_loader)

        print(
            f"[Localizer] Epoch {epoch+1} "
            f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            f"Train IoU: {train_iou:.4f} Val IoU: {val_iou:.4f} "
            f"Train Conf: {train_conf:.4f} Val Conf: {val_conf:.4f}"
        )

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/iou": train_iou,
            "val/iou": val_iou,
            "train/confidence": train_conf,
            "val/confidence": val_conf
        })

    torch.save(model.state_dict(), "localizer_image.pth")
    wandb.finish()


# ---------------------- SEGMENTER ----------------------
def train_segmenter():
    transfer_mode = "full"   # choose from: "strict", "partial", "full"

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

    def compute_pixel_accuracy(preds, masks):
        pred_classes = preds.argmax(dim=1)
        correct = (pred_classes == masks).sum().item()
        total = masks.numel()
        return correct / total

    def compute_dice_score(preds, masks, num_classes=3, eps=1e-6):
        pred_classes = preds.argmax(dim=1)
        dice_total = 0.0

        for cls in range(num_classes):
            pred_cls = (pred_classes == cls).float()
            true_cls = (masks == cls).float()

            intersection = (pred_cls * true_cls).sum()
            union = pred_cls.sum() + true_cls.sum()

            dice = (2 * intersection + eps) / (union + eps)
            dice_total += dice.item()

        return dice_total / num_classes

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_pixel_acc = 0
        train_dice = 0

        for i, batch in enumerate(train_loader):
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pixel_acc += compute_pixel_accuracy(preds, masks)
            train_dice += compute_dice_score(preds, masks)

            if i % 50 == 0:
                print(f"[Seg][{transfer_mode}][Epoch {epoch+1}] Batch {i}/{len(train_loader)} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_pixel_acc = 0
        val_dice = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                preds = model(imgs)
                loss = criterion(preds, masks)

                val_loss += loss.item()
                val_pixel_acc += compute_pixel_accuracy(preds, masks)
                val_dice += compute_dice_score(preds, masks)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_pixel_acc /= len(train_loader)
        val_pixel_acc /= len(val_loader)

        train_dice /= len(train_loader)
        val_dice /= len(val_loader)

        print(
            f"[Segmentation][{transfer_mode}] Epoch {epoch+1} "
            f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            f"Train Pixel Acc: {train_pixel_acc:.4f} Val Pixel Acc: {val_pixel_acc:.4f} "
            f"Train Dice: {train_dice:.4f} Val Dice: {val_dice:.4f}"
        )

        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/pixel_accuracy": train_pixel_acc,
            "val/pixel_accuracy": val_pixel_acc,
            "train/dice": train_dice,
            "val/dice": val_dice
        })

    torch.save(model.state_dict(), f"unet_image_{transfer_mode}.pth")
    wandb.finish()


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    train_classifier()  # run ONE at a time