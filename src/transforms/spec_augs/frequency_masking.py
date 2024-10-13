import random

import torchaudio
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param, prob, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.prob = prob

    def __call__(self, input: Tensor):
        use = random.random()
        return input if use > self.prob else self._aug(input.unsqueeze(1)).squeeze(1)
