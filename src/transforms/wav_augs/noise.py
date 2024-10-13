import torch_audiomentations
from torch import Tensor, nn


class Noise(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, input: Tensor):
        return self._aug(input.unsqueeze(1)).squeeze(1)
