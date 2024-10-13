import random

import torchaudio
from torch import Tensor, nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, prob, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param)
        self.prob = prob

    def __call__(self, input: Tensor):
        use = random.random()
        return input if use > self.prob else self._aug(input.unsqueeze(1)).squeeze(1)
