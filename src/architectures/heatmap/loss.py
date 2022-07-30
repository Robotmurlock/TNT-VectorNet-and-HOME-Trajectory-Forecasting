import torch
import torch.nn as nn


class PixelFocalLoss(nn.Module):
    def __init__(self):
        super(PixelFocalLoss, self).__init__()

    def forward(self, pred_heatmap: torch.Tensor, true_heatmap: torch.Tensor, da_area: torch.Tensor) -> torch.Tensor:
        mask = (true_heatmap == 1).float()
        pred_heatmap = torch.clamp(pred_heatmap, min=1e-3, max=1-1e-3)

        return -torch.mean(
            da_area * torch.pow(pred_heatmap - true_heatmap, 2) * (
                mask * torch.log(pred_heatmap)
                +
                (1-mask) * (torch.pow(1 - true_heatmap, 4) * torch.log(1 - pred_heatmap))
            )
        )
