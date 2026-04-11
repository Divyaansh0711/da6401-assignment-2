import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from models.localization import VGG11Localizer
from data.pets_dataset import OxfordPetsDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loader():
    dataset = OxfordPetsDataset(split="test")
    return DataLoader(dataset, batch_size=1, shuffle=False)


def tensor_to_pil(img_tensor):
    img = img_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def xywh_to_xyxy(box):
    x_center, y_center, w, h = box
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    return [x1, y1, x2, y2]


def clamp_box(box, img_size=224):
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_size - 1, x1))
    y1 = max(0, min(img_size - 1, y1))
    x2 = max(0, min(img_size - 1, x2))
    y2 = max(0, min(img_size - 1, y2))
    return [x1, y1, x2, y2]


def compute_iou(pred_box, gt_box, eps=1e-6):
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)

    union = pred_area + gt_area - inter_area + eps
    return inter_area / union


def draw_boxes(image_pil, gt_box, pred_box):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    draw.rectangle(gt_box, outline="green", width=3)
    draw.rectangle(pred_box, outline="red", width=3)

    return img


def main():
    wandb.init(
        project="da6401",
        name="localization_results_table",
        config={
            "model_path": "localizer_image.pth",
            "num_samples": 10,
            "gt_box_color": "green",
            "pred_box_color": "red",
        }
    )

    model = VGG11Localizer().to(DEVICE)
    model.load_state_dict(torch.load("localizer_image.pth", map_location=DEVICE))
    model.eval()

    loader = get_loader()

    table = wandb.Table(columns=[
        "sample_id",
        "image_with_boxes",
        "confidence_score",
        "iou",
    ])

    failure_case = None
    max_samples = 10

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_samples:
                break

            image = batch["image"].to(DEVICE)
            gt_bbox_xywh = batch["bbox"].squeeze(0).cpu().numpy()

            outputs = model(image)
            pred_bbox_xywh = outputs["bbox"].squeeze(0).cpu().numpy()
            confidence = outputs["confidence"].squeeze().item()

            gt_box = clamp_box(xywh_to_xyxy(gt_bbox_xywh))
            pred_box = clamp_box(xywh_to_xyxy(pred_bbox_xywh))

            iou = compute_iou(pred_box, gt_box)

            image_pil = tensor_to_pil(image)
            boxed_img = draw_boxes(image_pil, gt_box, pred_box)

            table.add_data(
                idx,
                wandb.Image(boxed_img, caption=f"Sample {idx}"),
                round(confidence, 4),
                round(iou, 4),
            )

            if failure_case is None:
                if confidence > 0.8 and iou < 0.3:
                    failure_case = {
                        "sample_id": idx,
                        "confidence": round(confidence, 4),
                        "iou": round(iou, 4),
                    }

    wandb.log({"localization_results_table": table})

    if failure_case is not None:
        wandb.log({
            "failure_case/sample_id": failure_case["sample_id"],
            "failure_case/confidence": failure_case["confidence"],
            "failure_case/iou": failure_case["iou"],
        })

    wandb.finish()


if __name__ == "__main__":
    main()