import torch_audiomentations
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(self, sample_rate, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(sample_rate, *args, **kwargs)

    def __call__(self, input: Tensor):
        return self._aug(input.unsqueeze(1)).squeeze(1)
