import random

import torchaudio
from torch import Tensor, nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param, prob, *args, **kwargs):
        self.aug_time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.prob = prob

    def __call__(self, input: Tensor):
        use = random.rand()
        return (
            input
            if use > self.prob
            else self.aug_time_mask(input.unsqueeze(1)).squeeze(1)
        )
