import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super(GCN, self).__init__()
        self._feature_encoding = nn.Linear(in_features, hidden_features)
        self._layernorm = nn.LayerNorm([hidden_features])
        self._relu = nn.ReLU()

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        enc = self._relu(self._layernorm(self._feature_encoding(features)))
        expanded_laplacian = adj.view(1, *adj.shape) \
            .expand(features.shape[0], *adj.shape)
        message = torch.bmm(enc, expanded_laplacian)
        aggregated_message = torch.max(message, dim=-2)[0].unsqueeze(-2).expand(*message.shape)
        return torch.concat([aggregated_message, enc], dim=-1)
