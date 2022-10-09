"""
VectorNet polyline encoder (subgraph)
"""
import numpy as np
import torch
import torch.nn as nn

from library.ml.building_blocks.graphs import GCN


class PGN(nn.Module):
    def __init__(self, cluster_size: int, in_features: int, n_layers: int):
        """
        Polyline encoder

        Args:
            cluster_size: Cluster size (polyline length)
            in_features: Number of features per polyline point
            n_layers: Number of layers
        """
        super(PGN, self).__init__()
        self._in_features = in_features
        self._out_features_list = [self._in_features * (2 ** i) for i in range(n_layers)]
        self._layers = nn.ModuleList([PolylineLayer(cluster_size, n) for n in self._out_features_list])

        # post
        self._post_linear = nn.Linear(2*self._out_features_list[-1], 2*self._out_features_list[-1])
        self._post_layernorm = nn.LayerNorm(2*self._out_features_list[-1])

    @property
    def out_features(self):
        """
        Returns: Output feature dimension
        """
        return self._in_features * (2 ** len(self._layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        x = torch.max(x, dim=-2)[0]
        x = self._post_layernorm(self._post_linear(x))
        return x


class PolylineLayer(nn.Module):
    def __init__(self, cluster_size: int, in_features: int):
        """
        Polyline Layer (GCN wrapper)

        Adjcency matrix:
        [
            [1, 1, 1, ..., 1]
            [0, 1, 1, ..., 1]
            ...
            [0, 0, ..., 0, 1]
        ]

        Args:
            cluster_size: Cluster size (polyline length)
            in_features: Number of features per polyline point
        """
        super(PolylineLayer, self).__init__()
        self._feature_encoding = nn.Linear(in_features, in_features)
        self._layernorm = nn.LayerNorm([in_features])
        self._gcn = GCN(in_features=in_features, hidden_features=in_features)

        raw_adj = np.array([[(1 if i >= j else 0) for i in range(cluster_size)] for j in range(cluster_size)])
        self._adj = nn.Parameter(torch.tensor(np.linalg.inv(np.diag(raw_adj.sum(axis=1))) @ raw_adj, dtype=torch.float32),
                                 requires_grad=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._feature_encoding(inputs)
        x = self._layernorm(x)
        x = self._gcn(x, self._adj)
        return x


def test():
    pgn = PGN(20, 9, 2).to('cuda')
    inputs = torch.randn(10, 200, 20, 9).to('cuda')
    print(pgn(inputs).shape)


if __name__ == '__main__':
    test()
