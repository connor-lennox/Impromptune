import torch.nn as nn


class Generator(nn.Module):
    """A top-level Generator class, defining the first half of the GAN structure.
    Since exact implementations of generation can differ, very little is present here. However,
    all Generators will have some amount of inputs and some amount of outputs.
    """

    def __init__(self, input_shape: tuple, output_shape: tuple):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
