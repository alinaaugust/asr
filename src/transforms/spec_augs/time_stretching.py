import random

import torchaudio
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, fixed_rate, prob, *args, **kwargs):
        self.time_stretching = torchaudio.transforms.TimeStretch(fixed_rate)
        self.prob = prob

    def __call__(self, input: Tensor):
        use = random.rand()
        return (
            input
            if use > self.prob
            else self.time_stretching(input.unsqueeze(1)).squeeze(1)
        )
