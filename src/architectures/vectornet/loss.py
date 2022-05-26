import torch
import torch.nn as nn
from typing import Tuple


class AnchorsLoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(AnchorsLoss, self).__init__()
        self.alpha = alpha
        self._ce = nn.CrossEntropyLoss()
        self._mse = nn.MSELoss()

    def forward(self, targets: torch.Tensor, confidences: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Finding the closest point
        distances = torch.sqrt((targets[..., 0] - ground_truth[..., 0]) ** 2 + (targets[..., 1] - ground_truth[..., 1]) ** 2)
        chosen_index = torch.argmin(distances, dim=-1)

        ce_loss = self._ce(confidences, chosen_index)  # closest target should have confidence 1 and all others should have 0
        mse_loss = self._mse(targets[..., chosen_index, :], ground_truth)  # MSE between the closest target and ground truth (end point)

        return ce_loss + self.alpha * mse_loss, ce_loss, mse_loss


def main():
    criteria = AnchorsLoss()
    anchors = torch.randn(1, 4, 2)
    confidences = torch.randn(1, 4)
    y = torch.randn(1, 1, 2)

    print(criteria(anchors, confidences, y))


if __name__ == '__main__':
    main()
