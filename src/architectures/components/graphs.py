import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super(GCN, self).__init__()
        self._feature_encoding = nn.Linear(in_features, hidden_features)
        self._layernorm = nn.LayerNorm([hidden_features])
        self._relu = nn.ReLU()

    def forward(self, features: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        enc = self._relu(self._layernorm(self._feature_encoding(features)))
        expanded_laplacian = laplacian.view(1, *laplacian.shape) \
            .expand(features.shape[0], *laplacian.shape)
        message = torch.bmm(expanded_laplacian, enc)
        return torch.concat([message, enc], dim=2)
