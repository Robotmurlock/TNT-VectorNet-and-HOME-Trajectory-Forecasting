import torch
import torch.nn as nn
from architectures.vectornet.polyline import PGN
from architectures.components.attention import SelfAttention
from typing import List


class VectorNet(nn.Module):
    def __init__(self, polyline_features: int):
        """
        Extracts context from all polyline information

        Args:
            polyline_features: Number of features per polyline point
        """
        super(VectorNet, self).__init__()
        self._polyline_encoder = PGN(in_features=polyline_features, n_layers=3)
        self._global = SelfAttention(in_features=self._polyline_encoder.out_features, projected_features=64)

        self._flatten = nn.Flatten(start_dim=0)
        self._linear = nn.Linear(64, 32)
        self._relu = nn.ReLU()

    def forward(self, polylines: List[torch.Tensor]) -> torch.Tensor:
        nodes = []
        for p_index in range(len(polylines)):
            nodes.append(self._polyline_encoder(polylines[p_index]))

        features = torch.stack(nodes)
        features = self._global(features)
        features = torch.max(self._relu(self._linear(features)), dim=0)[0]
        return features


def main():
    vn = VectorNet(polyline_features=10)
    polylines = [torch.randn(4, 10), torch.randn(20, 10), torch.randn(15, 10)]

    print(vn(polylines).shape)


if __name__ == '__main__':
    main()
