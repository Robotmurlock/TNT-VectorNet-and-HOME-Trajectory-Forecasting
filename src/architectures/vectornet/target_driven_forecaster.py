import os
import torch
from typing import Union, Tuple

from architectures.vectornet.target_generator import TargetGenerator
from architectures.vectornet.trajectory_forecaster import TrajectoryForecaster
from architectures.base import BaseModel


class TargetDrivenForecaster(BaseModel):
    def __init__(self, trajectory_length: int, polyline_features: int, n_trajectories: int, device: Union[str, torch.device]):
        super(TargetDrivenForecaster, self).__init__(trajectory_length=trajectory_length)
        self._n_trajectories = n_trajectories

        self._target_generator = TargetGenerator(polyline_features=polyline_features, device=device)
        self._trajectory_forecaster = TrajectoryForecaster(n_features=256, trajectory_length=trajectory_length)

    def forward(self, polylines: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, targets, confidences = self._target_generator(polylines, anchors)

        # choose top N trajectories
        filter_indexes = torch.argsort(confidences, descending=True)[:self._n_trajectories]
        filtered_targets = targets[filter_indexes]
        filtered_confidences = confidences[filter_indexes]

        trajectories = self._trajectory_forecaster(features, filtered_targets).cumsum(axis=1)
        return trajectories, filtered_confidences, filtered_targets

    def load_state(self, path: str) -> None:
        self._target_generator.load_state_dict(torch.load(os.path.join(path, 'target_generator.pt')))
        self._trajectory_forecaster.load_state_dict(torch.load(os.path.join(path, 'forecaster.pt')))


def test():
    polylines = torch.randn(200, 20, 9)
    anchors = torch.randn(75, 2)
    tdf = TargetDrivenForecaster(polyline_features=9, trajectory_length=20, n_trajectories=10, device='cpu')
    trajs, confs, targets = tdf(polylines, anchors)
    print(trajs.shape, confs.shape, targets.shape)


if __name__ == '__main__':
    test()
