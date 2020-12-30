import torch
import torch.nn as nn
import torch.nn.functional as func

from Generator import Generator


class DecoderStackGenerator(Generator):
    def __init__(self, d_model=128, n_head=8, num_layers=6, dropout=0):
        super().__init__(d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, n_head, dropout=dropout)
        self.decoder_stack = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, memory, predict_length, start_context=None):
        context = start_context or torch.zeros((1, memory.size()[1], self.d_model))
        out = None

        for _ in range(predict_length):
            out = self.decoder_stack(context, memory, tgt_mask=self.generate_square_subsequent_mask(context.size()[0]))
            last_elem = out[-1, :, :][None, :, :]
            context = torch.cat((context, last_elem))

        return out

    @staticmethod
    def generate_square_subsequent_mask(sz: int):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':
    gen = DecoderStackGenerator(d_model=8)
    noise = torch.randn((1, 1, 8))
    print(gen(noise, 5))