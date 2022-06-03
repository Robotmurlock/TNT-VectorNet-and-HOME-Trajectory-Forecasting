import torch
import torch.nn as nn
from architectures.vectornet.polyline import PGN
from architectures.components.attention import MultiHeadAttention
from typing import List, Union


class VectorNet(nn.Module):
    def __init__(self, polyline_features: int, device: Union[str, torch.device]):
        """
        Extracts context from all polyline information

        Args:
            polyline_features: Number of features per polyline point
        """
        super(VectorNet, self).__init__()
        self._polyline_encoder = PGN(in_features=polyline_features, n_layers=3, device=device)
        self._global = MultiHeadAttention(in_features=self._polyline_encoder.out_features, head_num=12, activation=nn.ReLU())

        self._flatten = nn.Flatten(start_dim=0)
        self._linear = nn.Linear(72, 128)
        self._relu = nn.ReLU()

    def forward(self, polylines: List[torch.Tensor]) -> torch.Tensor:
        features = self._polyline_encoder(polylines)
        features = self._global(features.unsqueeze(0)).squeeze(0)
        features = torch.max(self._relu(self._linear(features)), dim=0)[0]
        return features


def main():
    vn = VectorNet(polyline_features=9, device='cpu')
    polylines = torch.randn(200, 20, 9)
    print(vn(polylines).shape)


if __name__ == '__main__':
    main()
