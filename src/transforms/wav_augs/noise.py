import torch_audiomentations
from torch import Tensor, nn


class Noise(nn.Module):
    def __init__(self, *args, **kwargs):
        self.colored_noise = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, input: Tensor):
        return self.colored_noise(input.unsqueeze(1)).squeeze(1)
