import torch
import torch.nn as nn
import torch.nn.functional as F

from Predictive.Models.relative_multihead_attention import \
    PredictiveRelativeMultiheadAttention, EfficientRelativeMultiheadAttention, InformedPredictiveAttention, \
    LocalRelativeMultiheadAttention
from Predictive.Models.one_hot_embedding import OneHotEmbedding


class InformedTestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = OneHotEmbedding(240)
        # self.embedding = nn.Embedding(240, 240)
        # self.local_attn = LocalRelativeMultiheadAttention(embed_dim=240, key_dim=256, look_back=64, look_forward=64, value_dim=512)
        self.attn = InformedPredictiveAttention(240, 128, 240, 8, 128)
        # self.linear = nn.Linear(512, 240)
        # self.norm1 = nn.LayerNorm(240)

    def forward(self, xs):
        xs = self.embedding(xs)
        # xs = F.relu(self.local_attn(xs))
        xs = self.attn(xs)
        # xs = self.linear(xs)
        return xs


def generate_sequential_model():
    return nn.Sequential(
        OneHotEmbedding(240),
        EfficientRelativeMultiheadAttention(embed_dim=240, key_dim=128, value_dim=512, n_heads=8, relative_cutoff=128),
        nn.ReLU(),
        InformedPredictiveAttention(embed_dim=512, key_dim=128, value_dim=512, n_heads=8, relative_cutoff=128),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=240)
    )
