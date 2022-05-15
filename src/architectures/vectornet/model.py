import torch
import torch.nn as nn
from architectures.vectornet.polyline import PGN
from architectures.components.attention import SelfAttention
from typing import List


class TrajForecaster(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, trajectory_length: int):
        super(TrajForecaster, self).__init__()
        self._linear1 = nn.Linear(in_features, hidden_features)
        self._linear2 = nn.Linear(hidden_features, trajectory_length)
        self._relu = nn.ReLU()
        self._layernorm = nn.LayerNorm(hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear2(self._relu(self._layernorm(self._linear1(x))))


class VectorNet(nn.Module):
    def __init__(self, polyline_features: int, trajectory_length: int):
        super(VectorNet, self).__init__()
        self._polyline_encoder = PGN(in_features=polyline_features, n_layers=3)
        self._global = SelfAttention(in_features=self._polyline_encoder.out_features, projected_features=32)
        self._forecaster_x = TrajForecaster(in_features=16, hidden_features=32, trajectory_length=trajectory_length)
        self._forecaster_y = TrajForecaster(in_features=16, hidden_features=32, trajectory_length=trajectory_length)

        self._flatten = nn.Flatten(start_dim=0)
        self._linear = nn.Linear(32, 16)
        self._relu = nn.ReLU()

    def forward(self, polylines: List[torch.Tensor]) -> torch.Tensor:
        nodes = []
        for p_index in range(len(polylines)):
            nodes.append(self._polyline_encoder(polylines[p_index]))

        features = torch.stack(nodes)
        # print(features.shape)
        features = self._global(features)
        # print(features.shape)
        features = torch.max(self._relu(self._linear(features)), dim=0)[0]
        # print(features.shape)
        forecast_x, forecast_y = self._forecaster_x(features), self._forecaster_y(features)
        # print(features.shape)
        forecast = torch.stack([forecast_x, forecast_y], dim=1)
        # print(features.shape)

        return forecast


def main():
    vn = VectorNet(polyline_features=10, trajectory_length=5)
    polylines = [torch.randn(4, 10), torch.randn(20, 10), torch.randn(15, 10)]

    print(vn(polylines).shape)


if __name__ == '__main__':
    main()
