import torch.nn as nn
import torch
from typing import Union

from architectures.components.graphs import GCN


class PGN(nn.Module):
    def __init__(self, in_features: int, n_layers: int, device: Union[str, torch.device]):
        super(PGN, self).__init__()
        self._in_features = in_features
        self._out_features_list = [self._in_features * (2 ** i) for i in range(n_layers)]
        self._layers = nn.ModuleList([PolylineLayer(n, device) for n in self._out_features_list])

    @property
    def out_features(self):
        return self._in_features * (2 ** len(self._layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return torch.max(x, dim=-2)[0]


class PolylineLayer(nn.Module):
    def __init__(self, in_features: int, device: Union[str, torch.device]):
        super(PolylineLayer, self).__init__()
        self._feature_encoding = nn.Linear(in_features, in_features)
        self._layernorm = nn.LayerNorm([in_features])
        self._gcn = GCN(in_features=in_features, hidden_features=in_features)

        self._adj = torch.tensor([[(1 if i == j or i == j+1 else 0) for i in range(in_features)] for j in range(in_features)],
                                 dtype=torch.float32).to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._feature_encoding(inputs)
        x = self._layernorm(x)
        x = self._gcn(x, self._adj)
        return x


def test():
    pgn = PGN(9, 2, 'cpu')
    inputs = torch.randn(200, 20, 9)
    print(pgn(inputs).shape)


if __name__ == '__main__':
    test()
