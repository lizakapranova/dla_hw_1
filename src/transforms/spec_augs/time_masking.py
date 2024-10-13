import torchaudio.transforms
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._time_masking = torchaudio.transforms.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._time_masking(x).squeeze(1)
