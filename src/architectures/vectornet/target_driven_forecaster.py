import os
import torch
from typing import Union, Tuple

from architectures.vectornet.target_generator import TargetGenerator
from architectures.vectornet.trajectory_forecaster import TrajectoryForecaster
from architectures.base import BaseModel


class TargetDrivenForecaster(BaseModel):
    def __init__(self, cluster_size: int, trajectory_length: int, polyline_features: int, n_trajectories: int, device: Union[str, torch.device]):
        super(TargetDrivenForecaster, self).__init__(trajectory_length=trajectory_length)
        self._n_trajectories = n_trajectories

        self._target_generator = TargetGenerator(cluster_size=cluster_size, polyline_features=polyline_features, device=device)
        self._trajectory_forecaster = TrajectoryForecaster(n_features=256, trajectory_length=trajectory_length)

    def forward(self, polylines: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, offsets, confidences = self._target_generator(polylines, anchors)
        targets = anchors + offsets

        n_batches = features.shape[0]
        batch_filtered_anchors, batch_filtered_targets, batch_filtered_confidences = [], [], []
        for batch_index in range(n_batches):
            # choose top N targets
            instance_filter_indexes = torch.argsort(confidences[batch_index], descending=True)[:self._n_trajectories]
            instance_filtered_anchors = anchors[batch_index, instance_filter_indexes]
            instance_filtered_targets = targets[batch_index, instance_filter_indexes]
            instance_filtered_confidences = confidences[batch_index, instance_filter_indexes]

            batch_filtered_anchors.append(instance_filtered_anchors)
            batch_filtered_targets.append(instance_filtered_targets)
            batch_filtered_confidences.append(instance_filtered_confidences)

        filtered_anchors = torch.stack(batch_filtered_anchors)
        filtered_targets = torch.stack(batch_filtered_targets)
        filtered_confidences = torch.stack(batch_filtered_confidences)

        trajectories = self._trajectory_forecaster(features, filtered_targets).cumsum(axis=2)
        return trajectories, filtered_confidences, filtered_targets, filtered_anchors

    def load_state(self, path: str) -> None:
        self._target_generator.load_state_dict(torch.load(os.path.join(path, 'target_generator.pt')))
        self._trajectory_forecaster.load_state_dict(torch.load(os.path.join(path, 'forecaster.pt')))


def test():
    polylines = torch.randn(4, 200, 20, 14)
    anchors = torch.randn(4, 75, 2)
    tdf = TargetDrivenForecaster(cluster_size=20, polyline_features=14, trajectory_length=20, n_trajectories=10, device='cpu')
    trajs, confs, targets, anchors = tdf(polylines, anchors)
    print(trajs.shape, confs.shape, targets.shape)


if __name__ == '__main__':
    test()
