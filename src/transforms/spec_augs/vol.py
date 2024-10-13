import torchaudio.transforms
from torch import Tensor, nn


class Vol(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._vol = torchaudio.transforms.Vol(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._vol(x).squeeze(1)
