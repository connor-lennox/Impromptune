import torch
import torch.nn as nn
import torch.nn.functional as F

from Predictive.Models.relative_multihead_attention import \
    PredictiveRelativeMultiheadAttention, EfficientRelativeMultiheadAttention, InformedPredictiveAttention
from Predictive.Models.one_hot_embedding import OneHotEmbedding


class InformedTestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = OneHotEmbedding(333)
        self.attn = InformedPredictiveAttention(333, 64, 128, 8, 128)
        self.linear = nn.Linear(128, 333)

    def forward(self, xs):
        xs = self.embedding(xs)
        xs = self.attn(xs)
        xs = self.linear(xs)
        return xs
