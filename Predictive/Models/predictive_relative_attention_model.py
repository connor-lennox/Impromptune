import torch
import torch.nn as nn

from Predictive.Models.relative_multihead_attention import PredictiveRelativeMultiheadAttention


class PRAm(nn.Module):
    def __init__(self, embedding_dim=256, key_dim=64, value_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=333, embedding_dim=embedding_dim)
        self.pred_attn = PredictiveRelativeMultiheadAttention(embedding_dim, key_dim, value_dim)
        self.linear = nn.Linear(value_dim, 333)

    def forward(self, xs):
        xs = self.embedding(xs)
        xs = self.pred_attn(xs)
        xs = self.linear(xs)
        return xs


if __name__ == '__main__':
    test_pram = PRAm()
    test_input = torch.arange(0, 16)[None, :]
    test_result = test_pram(test_input)
    print(test_result.shape)
