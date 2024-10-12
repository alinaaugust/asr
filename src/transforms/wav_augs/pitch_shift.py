import torch_audiomentations
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        self.pitchshift = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, input: Tensor):
        return self.pitchshift(input.unsqueeze(1)).squeeze(1)
