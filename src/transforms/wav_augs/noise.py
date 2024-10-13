import random

import torch_audiomentations
from torch import Tensor, nn


class Noise(nn.Module):
    def __init__(self, prob, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)
        self.prob = prob

    def __call__(self, input: Tensor):
        use = random.random()
        return input if use > self.prob else self._aug(input.unsqueeze(1)).squeeze(1)
