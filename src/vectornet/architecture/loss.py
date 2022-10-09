"""
VectorNet loss functions
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TargetsLoss(nn.Module):
    def __init__(self, delta: float = 0.04):
        super(TargetsLoss, self).__init__()
        self._bce = nn.BCELoss(reduction='sum') # BinaryFocalLoss(alpha=4, reduction='sum')
        self._huber = nn.HuberLoss(delta=delta, reduction='sum')

    def forward(self, anchors: torch.Tensor, offsets: torch.Tensor, confidences: torch.Tensor, ground_truth: torch.Tensor) \
            -> Tuple[torch.Tensor, ...]:
        """
        Loss1 := CrossEntropyLoss(confidences, closest_target_index)
            Predicted target (anchor + offsets) that is the closest to ground truth point is correct class label

        Loss2 := HuberLoss(targets, ground_truth)
            Distance between the closest target point and ground truth point

        Args:
            anchors: Sampled agent end points by heuristic
            offsets: Anchor corrections
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
        # ideally the closest target should have confidence 1 and all others should have 0
        ce_loss = self._bce(F.softmax(confidences, dim=-1), closest_target_index_onehot)

        # offsets loss
        closest_anchors = torch.stack([anchors[index, closest_target_index[index], :] for index in range(anchors.shape[0])])
        closests_anchors_offsets = torch.stack([offsets[index, closest_target_index[index], :] for index in range(offsets.shape[0])])
        ground_truth_offset = ground_truth - closest_anchors

        huber_loss = self._huber(closests_anchors_offsets, ground_truth_offset)  # MSE between the closest target and ground truth (end point)

        # total loss
        total_loss = ce_loss + huber_loss

        return total_loss, ce_loss, huber_loss


class ForecastingLoss(nn.Module):
    def __init__(self, delta: float = 0.04):
        """
        Trajectory estimation loss

        Args:
            delta: Huber loss parameter
        """
        super(ForecastingLoss, self).__init__()
        self._huber = nn.HuberLoss(delta=delta, reduction='sum')

    def forward(self, forecasts: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        gt_traj_expanded = gt_traj.unsqueeze(1).repeat(1, forecasts.shape[1], 1, 1)
        return self._huber(forecasts, gt_traj_expanded)


class ForecastingScoringLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        """
        Trajectory confidence estimation loss
        """
        super(ForecastingScoringLoss, self).__init__()
        self._bce = nn.BCELoss(reduction='sum')
        self._temperature = temperature

    def forward(self, conf: torch.Tensor, forecasts: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
        gt_traj_expanded = gt_traj.unsqueeze(1).repeat(1, forecasts.shape[1], 1, 1)
        distances = -torch.max(torch.sqrt(
                torch.pow(forecasts[..., 0] - gt_traj_expanded[..., 0], 2)
                + torch.pow(forecasts[..., 1] - gt_traj_expanded[..., 1], 2)
            ), dim=-1
        )[0] / self._temperature
        # closest_target_index = torch.argmin(distances, dim=-1)
        # closest_target_index_onehot = F.one_hot(closest_target_index, num_classes=distances.shape[1]).float()
        gt_conf_proba = torch.clamp(F.softmax(distances, dim=-1), min=1e-3, max=1 - 1e-3)
        pred_conf_proba = torch.clamp(F.softmax(conf, dim=-1), min=1e-3, max=1 - 1e-3)
        # print(gt_conf_proba)
        # print(pred_conf_proba)

        # confidence loss
        # ideally the closest target should have confidence 1 and all others should have 0
        # return self._bce(F.softmax(conf, dim=-1), closest_target_index_onehot)
        return self._bce(pred_conf_proba, gt_conf_proba)


class LiteTNTLoss(nn.Module):
    def __init__(self, traj_scoring: bool = True, targets_delta: float = 0.04, forecasting_delta: float = 0.04):
        """
        Ensembled loss
        - Target end point estimation loss (offsets + confidence)
        - Trajectory estimation loss
        - Trejectory confidence (score) estimation loss (optional)

        Args:
            traj_scoring: Use traj scoring (confidence estimation)
            targets_delta: Target end point offsets huber loss parameter
            forecasting_delta: Trajectory huber loss parameter
        """
        super(LiteTNTLoss, self).__init__()
        self._traj_scoring_lambda = 0.1 if traj_scoring else 0.0
        self._tg_loss = TargetsLoss(delta=targets_delta)
        self._tf_loss = ForecastingLoss(delta=forecasting_delta)
        self._tfs_loss = ForecastingScoringLoss()

    def forward(
        self,
        anchors: torch.Tensor,
        offsets: torch.Tensor,
        confidences: torch.Tensor,
        ground_truth: torch.Tensor,
        forecasted_trajectories_gt_end_point: torch.Tensor,
        forecasted_trajectories: torch.Tensor,
        traj_conf: torch.Tensor,
        gt_traj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, tg_ce_loss, tg_huber_loss = self._tg_loss(anchors, offsets, confidences, ground_truth)
        tf_huber_loss = self._tf_loss(forecasted_trajectories_gt_end_point, gt_traj)
        tf_conf_loss = self._tfs_loss(traj_conf, forecasted_trajectories, gt_traj)

        total_loss = 0.1*tg_ce_loss + 0.1*tg_huber_loss + 1.0*tf_huber_loss + self._traj_scoring_lambda*tf_conf_loss
        return total_loss, tg_ce_loss, tg_huber_loss, tf_huber_loss, tf_conf_loss


def main():
    torch.set_printoptions(precision=13, sci_mode=False)

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

    # Test forecasting scoring loss
    fs_criteria = ForecastingScoringLoss()
    forecasts = torch.tensor([
        [[[0, 0], [1, 1.2], [2, 2.4]], [[0, 0], [1, 1], [20, 30]]],
        [[[0, 0], [2, 2], [4, 4]], [[0.1, 0.1], [2.1, 2.1], [4.1, 4.1]]]
    ], dtype=torch.float32)
    ground_truth = torch.tensor([
        [[0, 0], [1, 1], [2, 2]],
        [[0, 0], [2, 2], [4, 4]],
    ], dtype=torch.float32)
    confs = torch.tensor([
        [999, -999], [999, -999]
    ], dtype=torch.float32)

    print(fs_criteria(confs, forecasts, ground_truth))


if __name__ == '__main__':
    main()
