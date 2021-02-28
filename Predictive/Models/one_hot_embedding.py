import torch
import torch.nn as nn


class OneHotEmbedding(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.eye(num_embeddings), requires_grad=False)

    def forward(self, xs):
        return self.weights[xs]


class LockedEmbedding(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(embedding_matrix), requires_grad=False)

    def forward(self, xs):
        return self.weights[xs]
