import torch
import torch.nn as nn
import torch.nn.functional as F

from Predictive.Models.relative_multihead_attention import *


class GlobalRelativeAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256, n_heads=8, relative_cutoff=128, activation=F.relu):
        super().__init__()

        self.attn = EfficientRelativeMultiheadAttention(
            embedding_dim, key_dim=key_dim, value_dim=value_dim, n_heads=n_heads, relative_cutoff=relative_cutoff
        )
        self.norm1 = nn.LayerNorm(value_dim)
        self.linear = nn.Linear(value_dim, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.activation = activation

    def forward(self, xs):
        res = self.attn(xs)
        xs = self.norm1(xs + res)
        res = self.activation(self.linear(xs))
        xs = self.norm2(xs + res)
        return xs


class LocalRelativeAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256, n_heads=8, look_back=32, look_forward=32, activation=F.relu):
        super().__init__()

        self.attn = LocalRelativeMultiheadAttention(
            embedding_dim, key_dim=key_dim, value_dim=value_dim, n_heads=n_heads,
            look_back=look_back, look_forward=look_forward
        )
        self.norm1 = nn.LayerNorm(value_dim)
        self.linear = nn.Linear(value_dim, embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.activation = activation

    def forward(self, xs):
        res = self.attn(xs)
        xs = self.norm1(xs + res)
        res = self.activation(self.linear(xs))
        xs = self.norm2(xs + res)
        return xs