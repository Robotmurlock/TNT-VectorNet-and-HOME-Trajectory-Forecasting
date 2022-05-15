import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, in_features: int, projected_features: int):
        super(SelfAttention, self).__init__()
        self._d_scale = math.sqrt(projected_features)
        self._q_proj = nn.Linear(in_features, projected_features)
        self._k_proj = nn.Linear(in_features, projected_features)
        self._v_proj = nn.Linear(in_features, projected_features)

        self._softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._q_proj(x)
        key = self._k_proj(x)
        value = self._v_proj(x)

        weights = self._softmax(torch.mm(query, key.transpose(-2, -1)) / self._d_scale)
        return torch.mm(weights, value)


def test():
    inputs = torch.randn(3, 5, 10)
    att = SelfAttention(10, 4)

    print(att(inputs).shape)


if __name__ == '__main__':
    test()
