import torch
import torch.nn as nn

from Predictive.Models.relative_multihead_attention import \
    PredictiveRelativeMultiheadAttention, EfficientRelativeMultiheadAttention
from Predictive.Models.one_hot_embedding import OneHotEmbedding


class PRAm(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, use_onehot_embed=False, num_attn_layers=2):
        super().__init__()

        if use_onehot_embed:
            embedding_dim = 333
            self.embedding = OneHotEmbedding(num_embeddings=333)
        else:
            self.embedding = nn.Embedding(num_embeddings=333, embedding_dim=embedding_dim)

        self.rel_attn_layers = [EfficientRelativeMultiheadAttention(embedding_dim, key_dim, embedding_dim)
                                for _ in range(num_attn_layers)]
        self.pred_attn = PredictiveRelativeMultiheadAttention(embedding_dim, key_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 333)

    def forward(self, xs):
        xs = self.embedding(xs)
        for rel_attn in self.rel_attn_layers:
            xs = rel_attn(xs)
        xs = self.pred_attn(xs)
        xs = self.linear(xs)
        return xs


if __name__ == '__main__':
    test_pram = PRAm()
    test_input = torch.arange(0, 16)[None, :]
    test_result = test_pram(test_input)
    print(test_result.shape)
