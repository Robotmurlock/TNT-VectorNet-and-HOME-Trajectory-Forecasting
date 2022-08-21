import torch
import torch.nn as nn
from typing import Tuple, Dict


from architectures.heatmap import HeatmapModel, TorchModalitySampler, LightningTrajectoryForecaster


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

        self._target_sampler = TorchModalitySampler(n_targets=n_targets, radius=radius)

        # FIXME
        self._traj_features = traj_features
        self._traj_length = traj_length
        self._forecaster = None

    def forward(self, raster: torch.Tensor, agent_traj_hist: torch.Tensor, da_area: torch.Tensor) -> Dict[str, torch.Tensor]:
        heatmap = self._heatmap_estimator(raster, agent_traj_hist) * da_area
        targets = self._target_sampler(heatmap)
        forecasts = self._forecaster(agent_traj_hist, (targets - 112) / 25.0)
        return {
            'forecasts': forecasts,
            'targets': targets,
            'heatmap': heatmap
        }

    def load_weights(self, heatmap_estimator_path: str, trajectory_forecaster_path: str) -> None:
        self._heatmap_estimator.load_state_dict(torch.load(heatmap_estimator_path))

        # FIXME
        self._forecaster = LightningTrajectoryForecaster.load_from_checkpoint(
            checkpoint_path=trajectory_forecaster_path,
            traj_features=self._traj_features,
            traj_length=self._traj_length,
            train_config=None,
            in_features=3,
            trajectory_hist_length=20,
            trajectory_future_length=30
        )


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
