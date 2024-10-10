import re
from string import ascii_lowercase
from collections import defaultdict

import torch

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size=10, **kwargs):
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

        self.beam_size = beam_size

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
        prev_ind = -1
        for ind in inds:
            if ind != self.char2ind[self.EMPTY_TOK] and ind != prev_ind:
                decoded.append(self.ind2char[ind])
            prev_ind = ind
        return "".join(decoded)

    def ctc_decode_beam_search(self, probs) -> str:
        dp = {"": 1.0}
        is_prev_empty = False
        for prob in probs:
            dp, is_prev_empty = self.expand_end_merge_path(dp, prob, is_prev_empty)
            dp = self.truncate_paths(dp, self.beam_size)
        dp = [
            {"hypothesis": prefix, "probability": probability.item()}
            for prefix, probability in sorted(dp.items(), key=lambda x: x[1], reverse=True)
        ]
        return dp
    
    def expand_end_merge_path(self, dp, current_probs, is_prev_empty):
        #TODO: add annotation
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(current_probs):
            cur_char = self.ind2char[ind]
            for prefix, prev_probs in dp.items():
                last_char = prefix[-1] if prefix else ''
                if is_prev_empty or (not is_prev_empty and cur_char == last_char):
                    new_prefix = prefix
                else:
                    new_prefix = prefix + cur_char

                is_next_empty = cur_char == self.EMPTY_TOK
                new_dp[new_prefix] += prev_probs * next_token_prob
        return new_dp, is_next_empty

    def truncate_paths(self, dp):
        #TODO: add annotation
        sorted_dp = sorted(dp.items(), key=lambda x: x[1], reverse=True)
        truncated_dp = sorted_dp[: self.beam_size]
        return dict(truncated_dp)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
