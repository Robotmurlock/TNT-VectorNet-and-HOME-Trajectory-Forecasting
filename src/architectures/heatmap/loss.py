import torch
import torch.nn as nn


class PixelFocalLoss(nn.Module):
    def __init__(self):
        super(PixelFocalLoss, self).__init__()

    def forward(self, pred_heatmap: torch.Tensor, true_heatmap: torch.Tensor, da_area: torch.Tensor) -> torch.Tensor:
        pred_heatmap = torch.clamp(pred_heatmap, min=1e-6)
        return torch.mean(da_area * ((pred_heatmap - true_heatmap) ** 2))
