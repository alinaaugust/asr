import re
from collections import defaultdict
from string import ascii_lowercase

import kenlm
import torch
from pyctcdecode import build_ctcdecoder
from scipy.special import softmax


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(
        self,
        alphabet=None,
        use_lm=False,
        kenlm_path="saved/lowercase_3-gram.pruned.1e-7.arpa",
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.use_lm = use_lm

        if self.use_lm:
            lm_alphabet = ascii_lowercase + " "
            self.kenlm_path = kenlm_path
            self.beam_search_lm = build_ctcdecoder(
                [""] + [char for char in lm_alphabet],
                kenlm_model_path=self.kenlm_path,
                alpha=0.5,
                beta=0.2,
            )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char = self.EMPTY_TOK

        for ind in inds:
            current_char = self.ind2char[ind]

            if current_char == self.EMPTY_TOK:
                last_char = self.EMPTY_TOK
                continue

            if current_char != last_char:
                decoded.append(current_char)
                last_char = current_char
        return "".join(decoded)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def _expand_and_merge_path(self, state, next_token_probs):
        new_state = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), prob in state.items():
                if last_char == cur_char:
                    new_prefix = prefix
                elif cur_char != self.EMPTY_TOK:
                    new_prefix = prefix + cur_char
                else:
                    new_prefix = prefix
                new_state[(new_prefix, cur_char)] += prob * next_token_prob
        return new_state

    def _truncate_paths(self, state, beam_size=10):
        return dict(sorted(list(state.items()), key=lambda x: -x[1])[:beam_size])

    def _ctc_beam_search(self, probs, beam_size=10):
        state = {
            ("", self.EMPTY_TOK): 1.0,
        }

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().detach().numpy()
        probs = softmax(probs, axis=1)

        for prob in probs:
            state = self._expand_and_merge_path(state, prob)
            state = self._truncate_paths(state, beam_size)

        clean_result = defaultdict(float)
        for (prefix, last_char), prob in state.items():
            clean_sentence = (prefix + last_char).strip().replace(self.EMPTY_TOK, "")
            clean_result[clean_sentence] += prob

        clean_result = sorted(clean_result.items(), key=lambda x: -x[1])
        return [(sentence, prob) for sentence, prob in clean_result]

    def beam_search_result(self, probs, beam_size=10):
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().detach().numpy()
        if self.use_lm:
            return self.beam_search_lm.decode(softmax(probs, axis=1), beam_size)
        bs_result = self.text_encoder._ctc_beam_search(probs, beam_size)
        pred_text = bs_result[0][0]
        return pred_text
