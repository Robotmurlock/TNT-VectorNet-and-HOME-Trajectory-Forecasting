import torch
import torch.nn as nn
from architectures.vectornet.polyline import PGN
from architectures.components.attention import MultiHeadAttention
from typing import List, Union


class VectorNet(nn.Module):
    def __init__(self, cluster_size: int, polyline_features: int, device: Union[str, torch.device]):
        """
        Extracts context from all polyline information

        Args:
            polyline_features: Number of features per polyline point
        """
        super(VectorNet, self).__init__()
        self._polyline_encoder = PGN(cluster_size=cluster_size, in_features=polyline_features, n_layers=4, device=device)
        self._global = MultiHeadAttention(in_features=self._polyline_encoder.out_features, head_num=14, activation=nn.ReLU())

        self._linear = nn.Linear(224, 128)
        self._relu = nn.ReLU()

    def forward(self, polylines: List[torch.Tensor]) -> torch.Tensor:
        features = self._polyline_encoder(polylines)
        features = self._global(features)
        features = self._relu(self._linear(features))
        features = features[:, 0, :]
        return features


def main():
    vn = VectorNet(cluster_size=20, polyline_features=14, device='cpu')
    polylines = torch.randn(4, 200, 20, 14)
    print(vn(polylines).shape)


if __name__ == '__main__':
    main()
