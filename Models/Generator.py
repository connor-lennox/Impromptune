import torch.nn as nn


class Generator(nn.Module):
    """A top-level Generator class, defining the first half of the GAN structure.
    Since exact implementations of generation can differ, very little is present here. However,
    all Generators will have some amount of input features.
    """

    def __init__(self, d_model: int):
        super(Generator, self).__init__()
        self.d_model = d_model
