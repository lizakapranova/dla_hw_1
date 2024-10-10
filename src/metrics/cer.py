from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.text_encoder import CTCTextEncoder

# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)

class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder: CTCTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, probs: Tensor, probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        probs_numpy = probs.cpu().detach().numpy()
        lengths = probs_length.detach().numpy()
        for prob, length, target_text in zip(probs_numpy, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode_beam_search(prob[:length])[0]["hypothesis"]
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
