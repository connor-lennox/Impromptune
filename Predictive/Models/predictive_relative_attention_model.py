import torch
import torch.nn as nn

from Predictive.Models.relative_multihead_attention import PredictiveRelativeMultiheadAttention


class PRAm(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=333, embedding_dim=333)
        self.pred_attn = PredictiveRelativeMultiheadAttention(333, 64, 333)

    def forward(self, xs):
        xs = self.embedding(xs)
        xs = self.pred_attn(xs)
        return xs


if __name__ == '__main__':
    test_pram = PRAm()
    test_input = torch.arange(0, 16)[None, :]
    test_result = test_pram(test_input)
    print(test_result.shape)
