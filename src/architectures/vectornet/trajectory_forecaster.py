import torch
import torch.nn as nn


class TrajectoryForecaster(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        super(TrajectoryForecaster, self).__init__()
        self._n_features = n_features
        self._trajectory_length = trajectory_length

        self._linear = nn.Linear(n_features + 2, 128)
        self._l_trajectory_forecast = nn.Linear(130, 2 * self._trajectory_length)

        self._ln1 = nn.LayerNorm(128)
        self._lrelu = nn.LeakyReLU(0.1)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_targets = targets.shape[0]

        # merge features with targets
        expanded_features = features.unsqueeze(0).repeat(n_targets, 1)
        inputs = torch.concat([expanded_features, targets], dim=-1)

        out = self._lrelu(self._ln1(self._linear(inputs)))
        out = torch.concat([out, targets], dim=-1)  # skip connection

        trajectories = self._l_trajectory_forecast(out)
        trajectories = trajectories.view(*trajectories.shape[:-2], -1, self._trajectory_length, 2)
        return trajectories


def test():
    forecaster = TrajectoryForecaster(16, 10)
    features = torch.randn(16)
    targets = torch.randn(6, 2)
    trajectories = forecaster(features, targets)
    print(trajectories.shape)


if __name__ == '__main__':
    test()
