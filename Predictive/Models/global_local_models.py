import torch
import torch.nn as nn
import torch.nn.functional as F

from Predictive.Models.relative_multihead_attention import PredictiveRelativeMultiheadAttention
from Predictive.Models.relative_attention_blocks import LocalRelativeAttentionBlock, GlobalRelativeAttentionBlock
from Predictive.Models.one_hot_embedding import OneHotEmbedding


class StackedModel(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256, n_heads=8,
                 use_onehot_embed=False, local_range=(32, 32), relative_cutoff=128):
        super().__init__()

        self.key_dim = key_dim
        self.embedding_dim = embedding_dim
        self.value_dim = value_dim
        self.local_range = local_range
        self.relative_cutoff = relative_cutoff
        self.n_heads = n_heads
        self.use_onehot = use_onehot_embed

        if use_onehot_embed:
            embedding_dim = 240
            self.embedding_dim = 240
            self.embedding = OneHotEmbedding(num_embeddings=240)
        else:
            self.embedding = nn.Embedding(num_embeddings=240, embedding_dim=embedding_dim)

        self.local_attn_block = LocalRelativeAttentionBlock(embedding_dim=embedding_dim, key_dim=key_dim,
                                                            value_dim=value_dim, n_heads=n_heads,
                                                            look_back=local_range[0], look_forward=local_range[1])

        self.global_attn_block = GlobalRelativeAttentionBlock(embedding_dim=embedding_dim, key_dim=key_dim,
                                                              value_dim=value_dim, n_heads=n_heads,
                                                              relative_cutoff=relative_cutoff)

        self.predictive_layer = PredictiveRelativeMultiheadAttention(embedding_dim, key_dim=key_dim,
                                                                     value_dim=value_dim, n_heads=n_heads,
                                                                     relative_cutoff=relative_cutoff)

        self.linear = nn.Linear(embedding_dim, 240)

    def forward(self, xs):
        xs = self.embedding(xs)
        xs = self.local_attn_block(xs)
        xs = self.global_attn_block(xs)
        xs = self.predictive_layer(xs)
        xs = self.linear(xs)
        return xs


class ParallelModel(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256, n_heads=8,
                 use_onehot_embed=False, local_range=(32, 32), relative_cutoff=128):
        super().__init__()

        self.key_dim = key_dim
        self.embedding_dim = embedding_dim
        self.value_dim = value_dim
        self.local_range = local_range
        self.relative_cutoff = relative_cutoff
        self.n_heads = n_heads
        self.use_onehot = use_onehot_embed

        if use_onehot_embed:
            embedding_dim = 240
            self.embedding = OneHotEmbedding(num_embeddings=240)
        else:
            self.embedding = nn.Embedding(num_embeddings=240, embedding_dim=embedding_dim)

        self.local_attn_block = LocalRelativeAttentionBlock(embedding_dim=embedding_dim, key_dim=key_dim,
                                                            value_dim=value_dim, n_heads=n_heads,
                                                            look_back=local_range[0], look_forward=local_range[1])

        self.global_attn_block = GlobalRelativeAttentionBlock(embedding_dim=embedding_dim, key_dim=key_dim,
                                                              value_dim=value_dim, n_heads=n_heads,
                                                              relative_cutoff=relative_cutoff)

        self.predictive_layer = PredictiveRelativeMultiheadAttention(embedding_dim*2, key_dim=key_dim,
                                                                     value_dim=value_dim, n_heads=n_heads,
                                                                     relative_cutoff=relative_cutoff)

        self.linear = nn.Linear(embedding_dim, 240)

    def forward(self, xs):
        xs = self.embedding(xs)
        loc = self.local_attn_block(xs)
        glo = self.global_attn_block(xs)
        xs = torch.cat([loc, glo], dim=-1)
        xs = self.predictive_layer(xs)
        xs = self.linear(xs)
        return xs


if __name__ == '__main__':
    m = ParallelModel()
    test_input = torch.randint(0, high=240, size=(4, 8))
    test_output = m(test_input)
    print(test_output.shape)
