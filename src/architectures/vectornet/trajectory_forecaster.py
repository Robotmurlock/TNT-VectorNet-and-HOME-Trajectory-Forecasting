import torch
import torch.nn as nn


class TrajectoryForecaster(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        super(TrajectoryForecaster, self).__init__()
        self._n_features = n_features
        self._trajectory_length = trajectory_length

        self._linear = nn.Linear(n_features + 2, 64)
        self._l_trajectory_forecast = nn.Linear(66, 2 * self._trajectory_length)

        self._bn1 = nn.BatchNorm1d(64)
        self._lrelu = nn.LeakyReLU(0.1)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.concat([features, targets], dim=-1)
        out = self._lrelu(self._bn1(self._linear(inputs)))
        out = torch.concat([out, targets], dim=-1)  # skip connection

        trajectories = self._l_trajectory_forecast(out)
        trajectories = trajectories.view(*trajectories.shape[:-2], -1, self._trajectory_length, 2)
        return trajectories


def test():
    forecaster = TrajectoryForecaster(16, 10)
    features = torch.randn(6, 16)
    targets = torch.randn(6, 2)
    trajectories = forecaster(features, targets)
    print(trajectories.shape)


if __name__ == '__main__':
    test()
