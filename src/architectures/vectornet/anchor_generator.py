import torch
import torch.nn as nn
from typing import Tuple

from architectures.vectornet.context import VectorNet


class AnchorGenerator(nn.Module):
    def __init__(self, polyline_features: int):
        """
        Generates multiple targets from given anchor points sampled from centerlines

        Args:
            polyline_features: Number of features per polyline point
        """
        super(AnchorGenerator, self).__init__()
        self._vectornet = VectorNet(polyline_features)
        self._batch_norm = nn.BatchNorm1d(2)

        self._linear1 = nn.Linear(32, 600)
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
    ag = AnchorGenerator(polyline_features=10)
    polylines = [torch.randn(4, 10), torch.randn(20, 10), torch.randn(15, 10)]
    anchors = torch.randn(75, 2)

    targets, confidences = ag(polylines, anchors)
    print(targets.shape, confidences.shape)


if __name__ == '__main__':
    main()




