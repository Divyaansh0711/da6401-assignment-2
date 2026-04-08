import torch
from models.layers import CustomDropout

# Initialize
drop = CustomDropout(p=0.5)

# Input tensor
x = torch.ones(1000)

# Training mode
drop.train()
out = drop(x)
print("Train mean (should be ~1.0):", out.mean().item())

# Eval mode
drop.eval()
out_eval = drop(x)
print("Eval same as input:", torch.allclose(out_eval, x))