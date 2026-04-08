from models.vgg11 import VGG11Encoder
import torch

model = VGG11Encoder()
x = torch.randn(1, 3, 224, 224)

out, feats = model(x, return_features=True)

print(out.shape)        # should be [1, 512, 7, 7]
for k, v in feats.items():
    print(k, v.shape)