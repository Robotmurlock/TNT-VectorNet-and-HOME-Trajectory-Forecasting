import torch
import torch.nn as nn
from typing import Tuple


from architectures.heatmap import HeatmapModel, ModalitySampler, TrajectoryForecaster


class HeatmapTrajectoryForecaster(nn.Module):
    def __init__(
        self,
        encoder_input_shape: Tuple[int, int, int],
        decoder_input_shape: Tuple[int, int, int],
        traj_features: int,
        traj_length: int,

        n_targets: int,
        radius: int
    ):
        super().__init__()
        self._heatmap_estimator = HeatmapModel(
            encoder_input_shape=encoder_input_shape,
            decoder_input_shape=decoder_input_shape,
            traj_features=traj_features,
            traj_length=traj_length
        )

        self._target_sampler = ModalitySampler(n_targets=n_targets, radius=radius)

        self._forecaster = TrajectoryForecaster(in_features=traj_features, trajectory_future_length=traj_length)

    def forward(self, raster: torch.Tensor, agent_traj_hist: torch.Tensor) -> torch.Tensor:
        heatmap = self._heatmap_estimator(raster, agent_traj_hist)
        targets = self._target_sampler(heatmap)
        forecasts = self._forecaster(agent_traj_hist, targets)
        return forecasts

    def load_weights(self, heatmap_estimator_path: str, trajectory_forecater_path: str) -> None:
        self._heatmap_estimator.load_state_dict(torch.load(heatmap_estimator_path))
        self._forecaster.load_state_dict(torch.load(trajectory_forecater_path))


def test():
    raster = torch.randn(4, 48, 224, 224)
    trajectory = torch.rand(4, 20, 3)

    htf = HeatmapTrajectoryForecaster(
        encoder_input_shape=(48, 224, 224),
        decoder_input_shape=(512, 14, 14),
        traj_features=3,
        traj_length=20,

        n_targets=6,
        radius=2
    )

    outputs = htf(raster, trajectory)
    print(outputs.shape)


if __name__ == '__main__':
    test()
