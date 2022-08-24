import torch
import torch.nn as nn
from typing import Tuple, Dict


from architectures.heatmap import LightningHeatmapModel, TorchModalitySampler, LightningTrajectoryForecaster


class HeatmapTrajectoryForecaster(nn.Module):
    def __init__(
        self,
        encoder_input_shape: Tuple[int, int, int],
        decoder_input_shape: Tuple[int, int, int],
        traj_features: int,
        trajectory_history_window_length: int,
        trajectory_future_window_length: int,

        sampler_targets: int,
        sampler_radius: float,
        sampler_upscale: int,

        heatmap_estimator_path: str,
        trajectory_forecaster_path: str
    ):
        super().__init__()
        self._heatmap_estimator = LightningHeatmapModel.load_from_checkpoint(
            checkpoint_path=heatmap_estimator_path,
            encoder_input_shape=encoder_input_shape,
            decoder_input_shape=decoder_input_shape,
            traj_features=traj_features,
            traj_length=trajectory_history_window_length,
            sampler_radius=sampler_radius,
            sampler_targets=sampler_targets,
            base_lr=None,
            sched_step=None,
            sched_gamma=None
        )

        self._target_sampler = TorchModalitySampler(
            n_targets=sampler_targets,
            radius=sampler_radius,
            upscale=sampler_upscale
        )

        self._forecaster = self._forecaster = LightningTrajectoryForecaster.load_from_checkpoint(
            checkpoint_path=trajectory_forecaster_path,
            in_features=traj_features,
            trajectory_hist_length=trajectory_history_window_length,
            trajectory_future_length=trajectory_future_window_length,
            train_config=None
        )

    def forward(self, raster: torch.Tensor, agent_traj_hist: torch.Tensor, objects_traj_hists: torch.Tensor, da_area: torch.Tensor) \
            -> Dict[str, torch.Tensor]:
        heatmap = self._heatmap_estimator(raster, agent_traj_hist, objects_traj_hists) * da_area
        targets, confidences = self._target_sampler(heatmap)
        forecasts = self._forecaster(agent_traj_hist, (targets - 112) / 25.0)  # FIXME
        return {
            'forecasts': forecasts,
            'confidences': confidences,
            'targets': targets,
            'heatmap': heatmap
        }
