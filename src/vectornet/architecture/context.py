"""
Vectornet scene context feature extractor
"""
from typing import List

import torch
import torch.nn as nn

from library.ml.building_blocks.attention import MultiHeadAttention
from vectornet.architecture.polyline import PGN


class VectorNet(nn.Module):
    def __init__(self, cluster_size: int, polyline_features: int):
        """
        Extracts context from every polyline (trajectory)

        Args:
            polyline_features: Number of features per polyline point
        """
        super(VectorNet, self).__init__()
        self._polyline_encoder = PGN(cluster_size=cluster_size, in_features=polyline_features, n_layers=4)
        self._global = MultiHeadAttention(in_features=self._polyline_encoder.out_features, head_num=14, activation=nn.ReLU())

        self._linear = nn.Linear(448, 128)
        self._relu = nn.ReLU()

    def forward(self, polylines: List[torch.Tensor]) -> torch.Tensor:
        features = self._polyline_encoder(polylines)
        att_features = self._global(features)
        features = torch.concat([features, att_features], dim=-1)
        features = self._relu(self._linear(features))
        features = features[:, 0, :]
        return features


def main():
    vn = VectorNet(cluster_size=20, polyline_features=14)
    polylines = torch.randn(4, 200, 20, 14)
    print(vn(polylines).shape)


if __name__ == '__main__':
    main()
