import torch
import torch.nn as nn
from typing import Tuple, Union

from architectures.vectornet.context import VectorNet


class TargetGenerator(nn.Module):
    def __init__(self, polyline_features: int, device: Union[str, torch.device]):
        """
        Generates multiple targets from given anchor points sampled from centerlines

        Args:
            polyline_features: Number of features per polyline point
        """
        super(TargetGenerator, self).__init__()
        self._vectornet = VectorNet(polyline_features, device=device)
        self._batch_norm = nn.BatchNorm1d(2)

        self._linear1 = nn.Linear(128, 600)
        self._linear2 = nn.Linear(10, 16)
        self._l_corrections = nn.Linear(16, 2)
        self._l_confidence = nn.Linear(16, 1)
        self._relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor, anchors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._vectornet(inputs).unsqueeze(0)
        features = self._relu(self._linear1(features)).view(75, 8)
        features = self._relu(self._linear2(torch.concat([features, anchors], dim=-1)))

        # Outputs
        corrections = self._l_corrections(features)
        targets = anchors + corrections  # residual
        confidences = self._l_confidence(features).squeeze(-1)
        return features, targets, confidences


def main():
    tg = TargetGenerator(polyline_features=9, device='cpu')
    polylines = torch.randn(200, 20, 9)
    anchors = torch.randn(75, 2)

    features, targets, confidences = tg(polylines, anchors)
    print(features.shape, targets.shape, confidences.shape)


if __name__ == '__main__':
    main()
