"""
Basic Attention building blocks
"""
import torch
import torch.nn as nn
import math

from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features: int, head_num: int, activation: Optional[nn.Module] = None):
        """
        MultiHeadAttention layer (Attention is all you need)
        credit: https://github.com/CyberZHG/torch-multi-head-attention

        Args:
            in_features: Number of input features
            head_num: Number of attention heads
            activation: Activation function (optional)
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features` should be divisible by `head_num`!')
        self._head_in_features = in_features // head_num

        self._d_scale = math.sqrt(self._head_in_features)
        self._q_proj = nn.Linear(in_features, in_features)
        self._k_proj = nn.Linear(in_features, in_features)
        self._v_proj = nn.Linear(in_features, in_features)
        self._out_linear = nn.Linear(in_features, in_features)

        self._softmax = nn.Softmax(dim=-1)
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self._q_proj(x)
        key = self._k_proj(x)
        value = self._v_proj(x)

        if self._activation is not None:
            query = self._activation(query)
            key = self._activation(key)
            value = self._activation(value)

        query = self._reshape_to_batches(query)
        key = self._reshape_to_batches(key)
        value = self._reshape_to_batches(value)

        weights = self._softmax(torch.bmm(query, key.transpose(-2, -1)) / self._d_scale)
        features = torch.bmm(weights, value)
        # noinspection PyTypeChecker
        features = self._reshape_from_batches(features)

        features = self._out_linear(features)
        if self._activation:
            features = self._activation(features)

        return features

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self._head_in_features
        return x.reshape(batch_size, seq_len, self._head_in_features, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self._head_in_features, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self._head_in_features
        out_dim = in_feature * self._head_in_features
        return x.reshape(batch_size, self._head_in_features, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)


def test():
    inputs = torch.randn(4, 5, 12)
    att = MultiHeadAttention(12, 4, activation=nn.ReLU())

    print(att(inputs).shape)


if __name__ == '__main__':
    test()
