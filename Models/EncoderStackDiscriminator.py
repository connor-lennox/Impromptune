import torch
import torch.nn as nn
import torch.nn.functional as func

from Discriminator import Discriminator


class EncoderStackDiscriminator(Discriminator):
    def __init__(self, d_model=128, n_head=8, num_layers=6):
        super().__init__(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head)
        self.encoder_stack = nn.TransformerEncoder(encoder_layer, num_layers)

        # Linear layer for binary classification
        self.linear = nn.Linear(d_model, 2)

    def forward(self, src):
        out = self.encoder_stack(src)
        out = self.linear(out)
        out = torch.mean(out, dim=0, keepdim=False)
        out = func.softmax(out, 1)
        return out


if __name__ == '__main__':
    enc = EncoderStackDiscriminator(d_model=8)
    noise = torch.randn((2, 1, 8))
    result = enc(noise)
    print(result)
