import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader

from models.segmentation import VGG11UNet
from data.pets_dataset import OxfordPetsDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loader():
    dataset = OxfordPetsDataset(split="test")
    return DataLoader(dataset, batch_size=1, shuffle=False)


def tensor_to_image(img_tensor):
    img = img_tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img


def mask_to_image(mask_array):
    # simple grayscale visualization for trimap classes 0,1,2
    mask_vis = (mask_array * 127).astype(np.uint8)
    return mask_vis


def main():
    wandb.init(
        project="da6401",
        name="segmentation_visuals_table",
        config={
            "models": [
                "unet_image_strict.pth",
                "unet_image_partial.pth",
                "unet_image_full.pth",
            ]
        }
    )

    model_paths = {
        "strict": "unet_image_strict.pth",
        "partial": "unet_image_partial.pth",
        "full": "unet_image_full.pth",
    }

    models = {}
    for mode, path in model_paths.items():
        model = VGG11UNet(num_classes=3).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        models[mode] = model

    loader = get_loader()

    table = wandb.Table(columns=[
        "sample_id",
        "original_image",
        "ground_truth_trimap",
        "predicted_trimap_strict",
        "predicted_trimap_partial",
        "predicted_trimap_full",
    ])

    max_samples = 5

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_samples:
                break

            image = batch["image"].to(DEVICE)
            true_mask = batch["mask"].squeeze(0).cpu().numpy().astype(np.uint8)
            image_np = tensor_to_image(image)
            true_mask_img = mask_to_image(true_mask)

            pred_images = {}

            for mode, model in models.items():
                preds = model(image)
                pred_mask = preds.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                pred_images[mode] = mask_to_image(pred_mask)

            table.add_data(
                idx,
                wandb.Image(image_np, caption=f"Original {idx}"),
                wandb.Image(true_mask_img, caption=f"GT Trimap {idx}"),
                wandb.Image(pred_images["strict"], caption=f"Strict Pred {idx}"),
                wandb.Image(pred_images["partial"], caption=f"Partial Pred {idx}"),
                wandb.Image(pred_images["full"], caption=f"Full Pred {idx}"),
            )

    wandb.log({"segmentation_results_table": table})
    wandb.finish()


if __name__ == "__main__":
    main()