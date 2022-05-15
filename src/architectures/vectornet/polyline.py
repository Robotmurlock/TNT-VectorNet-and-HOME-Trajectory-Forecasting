import torch.nn as nn
import torch


class PGN(nn.Module):
    def __init__(self, in_features: int, n_layers: int):
        super(PGN, self).__init__()
        self._in_features = in_features
        self._out_features_list = [self._in_features * (2 ** i) for i in range(n_layers)]
        self._layers = nn.ModuleList([PolylineLayer(n) for n in self._out_features_list])

    @property
    def out_features(self):
        return self._in_features * (2 ** len(self._layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return torch.mean(x, dim=-2)


class PolylineLayer(nn.Module):
    def __init__(self, in_features: int):
        super(PolylineLayer, self).__init__()
        self._feature_encoding = nn.Linear(in_features, in_features)
        self._layernorm = nn.LayerNorm([in_features])
        self._relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._feature_encoding(inputs)
        x = self._layernorm(x)
        x = self._relu(x)
        return torch.concat([inputs, x], dim=-1)


def test():
    pgn = PGN(10, 2)

    inputs = torch.randn(1, 4, 10)
    print(pgn(inputs).shape)


if __name__ == '__main__':
    test()
