from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(
        self,
        text_encoder: CTCTextEncoder,
        beam_size=10,
        lm_usage=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        self.lm_usage = lm_usage

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        # if self.lm_usage:
        #     fmdkfmd

        cers = []
        preds = log_probs.cpu().detach().numpy()
        lengths = log_probs_length.cpu().detach().numpy()

        for log_prob, length, target in zip(preds, lengths, text):
            target = self.text_encoder.normalize_text(target)

            bs_result = self.text_encoder.ctc_beam_search(
                log_prob[:length], self.beam_size
            )
            pred_text = bs_result[0][0]

            cers.append(calc_cer(target, pred_text))
        return sum(cers) / len(cers)
