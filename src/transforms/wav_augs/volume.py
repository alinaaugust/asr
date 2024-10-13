import torchaudio
from torch import Tensor, nn


class Volume(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.Vol(*args, **kwargs)

    def __call__(self, input: Tensor):
        return self._aug(input.unsqueeze(1)).squeeze(1)
