import torch
from models.multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel()
x = torch.randn(1, 3, 224, 224)

out = model(x)

print(out["classification"].shape)  # [1, 37]
print(out["localization"].shape)   # [1, 4]
print(out["segmentation"].shape)   # [1, 3, 224, 224]