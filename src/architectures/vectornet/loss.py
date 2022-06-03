import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AnchorsLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, delta: float = 0.16):
        super(AnchorsLoss, self).__init__()
        self.alpha = alpha
        self._ce = nn.CrossEntropyLoss()
        self._huber = nn.HuberLoss(delta=delta)

    def forward(self, targets: torch.Tensor, confidences: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Loss1 := CrossEntropyLoss(confidences, closest_target_index)
            Predicted target (anchor + offsets) that is the closest to ground truth point is correct class label

        Loss2 := HuberLoss(targets, ground_truth)
            Distance between the closest target point and ground truth point

        Args:
            targets: Predicted targets (future trajectory last point) - anchor + offsets
            confidences: Confidence (probability) for each target
            ground_truth: Ground truth target

        Returns: Loss1 + alpha * Loss2
        """
        # Finding the closest point
        distances = torch.sqrt((targets[..., 0] - ground_truth[..., 0]) ** 2 + (targets[..., 1] - ground_truth[..., 1]) ** 2)
        closest_target_index = torch.argmin(distances, dim=-1)

        ce_loss = self._ce(confidences, closest_target_index)  # closest target should have confidence 1 and all others should have 0
        huber_loss = self._huber(targets[..., closest_target_index, :], ground_truth)  # MSE between the closest target and ground truth (end point)
        total_loss = ce_loss + self.alpha * huber_loss

        return total_loss, ce_loss, huber_loss


class ForecastingLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, delta: float = 0.16):
        super(ForecastingLoss, self).__init__()
        self.alpha = alpha
        self._ce = nn.CrossEntropyLoss()
        self._huber = nn.HuberLoss(delta=delta)

        self._softmax = nn.Softmax(dim=-1)

    def forward(self, forecasts: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        gt_traj_expanded = gt_traj.unsqueeze(0).expand(forecasts.shape[0], gt_traj.shape[-2], 2)
        huber_loss = self._huber(forecasts, gt_traj_expanded)
        return huber_loss


def main():
    criteria = AnchorsLoss()
    anchors = torch.randn(1, 4, 2)
    confidences = torch.randn(1, 4)
    y = torch.randn(1, 1, 2)

    print(criteria(anchors, confidences, y))


if __name__ == '__main__':
    main()
