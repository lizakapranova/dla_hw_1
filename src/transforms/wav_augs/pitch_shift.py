import torch_audiomentations
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._pitch_shift = torch_audiomentations.PitchShift(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._pitch_shift(x).squeeze(1)
