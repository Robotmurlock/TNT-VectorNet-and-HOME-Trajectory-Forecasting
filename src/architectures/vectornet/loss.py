import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TargetsLoss(nn.Module):
    def __init__(self, delta: float = 0.16):
        super(TargetsLoss, self).__init__()
        self._bce = nn.BCEWithLogitsLoss()
        self._huber = nn.HuberLoss(delta=delta, reduction='mean')


    def forward(self, anchors: torch.Tensor, offsets: torch.Tensor, confidences: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Loss1 := CrossEntropyLoss(confidences, closest_target_index)
            Predicted target (anchor + offsets) that is the closest to ground truth point is correct class label

        Loss2 := HuberLoss(targets, ground_truth)
            Distance between the closest target point and ground truth point

        Args:
            anchors: TODO
            offsets: TODO
            confidences: Confidence (probability) for each target
            ground_truth: Ground truth target

        Returns: Loss1 + alpha * Loss2
        """
        # Finding the closest point
        ground_truth_expanded = ground_truth.unsqueeze(1).repeat(1, anchors.shape[1], 1)
        distances = torch.sqrt((anchors[..., 0] - ground_truth_expanded[..., 0]) ** 2 + (anchors[..., 1] - ground_truth_expanded[..., 1]) ** 2)
        closest_target_index = torch.argmin(distances, dim=-1)
        closest_target_index_onehot = F.one_hot(closest_target_index, num_classes=distances.shape[1]).float()

        # confidence loss
        ce_loss = self._bce(confidences, closest_target_index_onehot)  # closest target should have confidence 1 and all others should have 0

        # offsets loss
        closest_anchors = torch.stack([anchors[index, closest_target_index[index], :] for index in range(anchors.shape[0])])
        closests_anchors_offsets = torch.stack([offsets[index, closest_target_index[index], :] for index in range(offsets.shape[0])])
        ground_truth_offset = ground_truth - closest_anchors

        huber_loss = self._huber(closests_anchors_offsets, ground_truth_offset)  # MSE between the closest target and ground truth (end point)

        # total loss
        total_loss = ce_loss + huber_loss

        return total_loss, ce_loss, huber_loss


class ForecastingLoss(nn.Module):
    def __init__(self, delta: float = 0.16):
        super(ForecastingLoss, self).__init__()
        self._huber = nn.HuberLoss(delta=delta)

    def forward(self, forecasts: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        gt_traj_expanded = gt_traj.unsqueeze(1).repeat(1, forecasts.shape[1], 1, 1)
        return self._huber(forecasts, gt_traj_expanded)


class LiteTNTLoss(nn.Module):
    def __init__(self, targets_delta: float = 0.16, forecasting_delta: float = 0.16):
        super(LiteTNTLoss, self).__init__()
        self._tg_loss = TargetsLoss(delta=targets_delta)
        self._tf_loss = ForecastingLoss(delta=forecasting_delta)

    def forward(
        self,
        anchors: torch.Tensor,
        offsets: torch.Tensor,
        confidences: torch.Tensor,
        ground_truth: torch.Tensor,
        forecasts: torch.Tensor,
        gt_traj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, tg_ce_loss, tg_huber_loss = self._tg_loss(anchors, offsets, confidences, ground_truth)
        tf_huber_loss = self._tf_loss(forecasts, gt_traj)

        total_loss = 0.1*tg_ce_loss + 0.1*tg_huber_loss + 1.0*tf_huber_loss
        return total_loss, tg_ce_loss, tg_huber_loss, tg_huber_loss


def main():
    # Test targets loss
    t_criteria = TargetsLoss()
    anchors = torch.tensor([
        [[0, 1], [0, 0]],
        [[0, 0.1], [0, 0]],
    ], dtype=torch.float32)
    offsets = torch.tensor([
        [[0, 0.1], [0, 0]],
        [[0, 0.2], [0, 0]],
    ], dtype=torch.float32)
    confidences = torch.tensor([
        [100.0, 0],
        [100, 0.1]
    ])
    gt_pos = torch.tensor([
        [0, 1.1],
        [0, 0.3],
    ], dtype=torch.float32)

    print(t_criteria(anchors, offsets, confidences, gt_pos))

    # Test forecasting loss
    f_criteria = ForecastingLoss()
    forecasts = torch.tensor([
        [[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]],
        [[[0, 0], [2, 2], [4, 4]], [[0, 0], [2, 2], [4, 4]]]
    ], dtype=torch.float32)
    ground_truth = torch.tensor([
        [[0, 0], [1, 1], [2, 2]],
        [[0, 0], [2, 2], [4, 4]],
    ], dtype=torch.float32)

    print(f_criteria(forecasts, ground_truth))


if __name__ == '__main__':
    main()
