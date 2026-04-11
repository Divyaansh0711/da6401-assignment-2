import torch
import wandb

from torch.utils.data import DataLoader

from models.classification import VGG11Classifier
from data.pets_dataset import OxfordPetsDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_one_sample():
    dataset = OxfordPetsDataset(split="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    image = batch["image"].to(DEVICE)
    label = batch["label"].to(DEVICE)
    return image, label


def main():
    wandb.init(
        project="da6401",
        name="bn_activation_analysis",
        config={
            "analysis": "conv3_activation_distribution",
            "models_compared": ["classifier_bn_True.pth", "classifier_bn_False.pth"],
        },
    )

    # Load both models
    model_bn = VGG11Classifier(dropout_p=0.2, use_batchnorm=True).to(DEVICE)
    model_no_bn = VGG11Classifier(dropout_p=0.2, use_batchnorm=False).to(DEVICE)

    model_bn.load_state_dict(torch.load("classifier_bn_True.pth", map_location=DEVICE))
    model_no_bn.load_state_dict(torch.load("classifier_bn_False.pth", map_location=DEVICE))

    model_bn.eval()
    model_no_bn.eval()

    # Storage for activations
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    # Hook on block3 output
    hook_bn = model_bn.encoder.block3.register_forward_hook(get_hook("bn_conv3"))
    hook_no_bn = model_no_bn.encoder.block3.register_forward_hook(get_hook("no_bn_conv3"))

    # Get same sample image
    image, label = get_one_sample()

    with torch.no_grad():
        out_bn = model_bn(image)
        out_no_bn = model_no_bn(image)

    # Remove hooks
    hook_bn.remove()
    hook_no_bn.remove()

    bn_act = activations["bn_conv3"].flatten().numpy()
    no_bn_act = activations["no_bn_conv3"].flatten().numpy()

    pred_bn = out_bn.argmax(dim=1).item()
    pred_no_bn = out_no_bn.argmax(dim=1).item()
    true_label = label.item()

    print("BN activation shape:", activations["bn_conv3"].shape)
    print("No BN activation shape:", activations["no_bn_conv3"].shape)

    print("BN min/max:", bn_act.min(), bn_act.max())
    print("No BN min/max:", no_bn_act.min(), no_bn_act.max())

    print("BN mean/std:", bn_act.mean(), bn_act.std())
    print("No BN mean/std:", no_bn_act.mean(), no_bn_act.std())

    wandb.log({
        "activations/bn_conv3_hist": wandb.Histogram(bn_act),
        "activations/no_bn_conv3_hist": wandb.Histogram(no_bn_act),
        "sample/true_label": true_label,
        "sample/pred_bn": pred_bn,
        "sample/pred_no_bn": pred_no_bn,
    })

    wandb.finish()


if __name__ == "__main__":
    main()