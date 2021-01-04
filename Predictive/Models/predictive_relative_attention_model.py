import torch
import torch.nn as nn
import torch.nn.functional as F

from Predictive.Models.relative_multihead_attention import \
    PredictiveRelativeMultiheadAttention, EfficientRelativeMultiheadAttention
from Predictive.Models.one_hot_embedding import OneHotEmbedding


class PRAm(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256, use_onehot_embed=False, num_attn_layers=2, relative_cutoff=128):
        super().__init__()

        if use_onehot_embed:
            embedding_dim = 333
            self.embedding = OneHotEmbedding(num_embeddings=333)
        else:
            self.embedding = nn.Embedding(num_embeddings=333, embedding_dim=embedding_dim)

        self.rel_attn_layers = nn.ModuleList([PRAmBlock(embedding_dim, key_dim, value_dim, relative_cutoff)
                                              for _ in range(num_attn_layers)])
        self.pred_attn = PredictiveRelativeMultiheadAttention(
            embedding_dim, key_dim, value_dim, relative_cutoff=relative_cutoff
        )
        self.linear = nn.Linear(value_dim, 333)

    def forward(self, xs):
        xs = self.embedding(xs)
        for rel_attn in self.rel_attn_layers:
            xs = rel_attn(xs)
        xs = self.pred_attn(xs)
        xs = self.linear(xs)
        return xs


class PRAmBlock(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256, relative_cutoff=128, activation=F.relu):
        super().__init__()

        self.attn = EfficientRelativeMultiheadAttention(
            embedding_dim, key_dim=key_dim, value_dim=value_dim, relative_cutoff=relative_cutoff
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


if __name__ == '__main__':
    test_pram = PRAm()
    test_input = torch.arange(0, 16)[None, :]
    test_result = test_pram(test_input)
    print(test_result.shape)
