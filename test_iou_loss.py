import torch
from losses.iou_loss import IoULoss

loss_fn = IoULoss()

pred = torch.tensor([[50., 50., 20., 20.]])
target = torch.tensor([[50., 50., 20., 20.]])

loss = loss_fn(pred, target)
print("Loss (should be 0):", loss.item())