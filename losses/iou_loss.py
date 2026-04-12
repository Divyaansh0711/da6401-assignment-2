"""Custom IoU loss 
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        """
        super().__init__()
        self.eps=eps

        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss."""

        #convert (x_center, y_center, w, h) → (x1, y1, x2, y2)
        pred_x1=pred_boxes[:, 0]-pred_boxes[:, 2]/2
        pred_y1=pred_boxes[:, 1]-pred_boxes[:, 3]/2
        pred_x2=pred_boxes[:, 0]+pred_boxes[:, 2]/2
        pred_y2=pred_boxes[:, 1]+pred_boxes[:, 3]/2

        target_x1=target_boxes[:, 0]-target_boxes[:, 2]/2
        target_y1=target_boxes[:, 1]-target_boxes[:, 3]/2
        target_x2=target_boxes[:, 0]+target_boxes[:, 2]/2
        target_y2=target_boxes[:, 1]+target_boxes[:, 3]/2

        #intersection
        inter_x1=torch.max(pred_x1, target_x1)
        inter_y1=torch.max(pred_y1, target_y1)
        inter_x2=torch.min(pred_x2, target_x2)
        inter_y2=torch.min(pred_y2, target_y2)

        inter_w=(inter_x2 - inter_x1).clamp(min=0)
        inter_h=(inter_y2 - inter_y1).clamp(min=0)
        inter_area=inter_w * inter_h

        #areas
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)

        #union
        union=pred_area+target_area-inter_area+self.eps

        #IoU
        iou=inter_area/union
        iou=iou.clamp(min=0.0, max=1.0)

        #loss
        loss=1-iou

        #reduction
        if self.reduction=="mean":
            return loss.mean()
        elif self.reduction=="sum":
            return loss.sum()
        else:
            return loss