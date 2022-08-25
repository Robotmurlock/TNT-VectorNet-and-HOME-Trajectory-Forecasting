import torch
import torch.nn as nn


class TrajectoryForecaster(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        """
        Estimates trajectory given end point and VectorNet features

        Args:
            n_features: Vectornet features dimension
            trajectory_length: Output trajectory length
        """
        super(TrajectoryForecaster, self).__init__()
        self._n_features = n_features
        self._trajectory_length = trajectory_length

        self._linear = nn.Linear(n_features + 2, 128)
        self._l_trajectory_forecast = nn.Linear(130, 2 * self._trajectory_length)

        self._ln1 = nn.LayerNorm(128)
        self._lrelu = nn.LeakyReLU(0.1)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_targets = targets.shape[1]

        # merge features with targets
        expanded_features = features.unsqueeze(1).repeat(1, n_targets, 1)
        inputs = torch.concat([expanded_features, targets], dim=-1)

        out = self._lrelu(self._ln1(self._linear(inputs)))
        out = torch.concat([out, targets], dim=-1)  # skip connection

        trajectories = self._l_trajectory_forecast(out)
        trajectories = trajectories.view(*trajectories.shape[:-2], -1, self._trajectory_length, 2)
        return trajectories


class TrajectoryScorer(nn.Module):
    def __init__(self, n_features: int, trajectory_length: int):
        super(TrajectoryScorer, self).__init__()
        self._n_features = n_features
        self._trajectory_length = trajectory_length

        self._linear = nn.Linear(n_features + 2 * trajectory_length, 128)
        self._l_trajectory_score = nn.Linear(128 + 2 * trajectory_length, 1)

        self._ln1 = nn.LayerNorm(128)
        self._lrelu = nn.LeakyReLU(0.1)

    def forward(self, features: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        n_trajectories = trajectories.shape[1]

        # merge features with targets
        expanded_features = features.unsqueeze(1).repeat(1, n_trajectories, 1)
        flatten_trajectories = trajectories.view(*trajectories.shape[:2], -1)
        inputs = torch.concat([expanded_features, flatten_trajectories], dim=-1)

        out = self._lrelu(self._ln1(self._linear(inputs)))
        out = torch.concat([out, flatten_trajectories], dim=-1)  # skip connection

        out = self._l_trajectory_score(out)
        scores = out.view(*out.shape[:-2], -1, 1).squeeze(-1)
        return scores


def test():
    forecaster = TrajectoryForecaster(16, 20)
    scorer = TrajectoryScorer(16, 20)
    features = torch.randn(4, 16)
    targets = torch.randn(4, 6, 2)
    trajectories = forecaster(features, targets)
    scores = scorer(features, trajectories)
    print(trajectories.shape, scores.shape)


if __name__ == '__main__':
    test()
