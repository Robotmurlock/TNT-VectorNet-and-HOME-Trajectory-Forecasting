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
        enc_flatten = enc.view(-1, *enc.shape[-2:])
        expanded_laplacian = adj.view(1, *adj.shape) \
            .expand(enc_flatten.shape[0], *adj.shape)
        message_flatten = torch.bmm(expanded_laplacian, enc_flatten)
        message = message_flatten.view(enc.shape)
        aggregated_message = torch.max(message, dim=-2)[0].unsqueeze(-2).expand(*message.shape)
        return torch.concat([aggregated_message, enc], dim=-1)


def test():
    gcn = GCN(10, 20)
    inputs = torch.randn(4, 10, 3, 10)
    adj = torch.tensor([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1]
    ], dtype=torch.float32)

    print(gcn(inputs, adj).shape)


if __name__ == '__main__':
    test()
