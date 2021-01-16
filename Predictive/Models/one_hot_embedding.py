import torch
import torch.nn as nn


class OneHotEmbedding(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.eye(num_embeddings), requires_grad=False)

    def forward(self, xs):
        return self.weights[xs]
