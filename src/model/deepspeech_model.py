import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, module):
        super(ConvBlock, self).__init__()
        self.module = module

    def forward(self, x, length):
        for module in self.module:
            x = module(x)
            length = self.transform_input_lengths(length, module)
            b, c, f, t = x.shape
            mask = (torch.arange(t).tile((b, c, f, 1)) >= length[:, None, None, None]).to(x.device)
            x = x.masked_fill(mask, 0)
        return x, length

    @staticmethod
    def transform_input_lengths(length, module):
        if type(module) is not nn.Conv2d:
            return length
        else:
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            return (length + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1


class RNNBlock(nn.Module):
    """
    Class for RNN layer.
    """

    def __init__(self, input_size, hidden_size, batch_first=True, dropout=0.0, batch_norm=True):
        super(RNNBlock, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first, dropout=dropout)
        self.bn = nn.BatchNorm1d(input_size) if batch_norm else None

    def forward(self, x, length):
        if self.bn is not None:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2).contiguous()  # B x T x F
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.rnn(x, None)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x

class DeepSpeech2(nn.Module):
    """
    Implementation is based on https://arxiv.org/abs/1512.02595.
    """

    def __init__(self, n_feats, n_tokens, fc_hidden, rnn_layers, dropout):
        super(DeepSpeech2, self).__init__()
        self.conv = ConvBlock(
            nn.Sequential(
                nn.Conv2d(1, 32, (41, 11), stride=(2, 2), padding=(20, 5), bias=False),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 96, (21, 11), stride=(2, 1), padding=(10, 5), bias=False),
            )
        )

        input_size = n_feats
        input_size = (input_size + 20 * 2 - 41) // 2 + 1
        input_size = (input_size + 10 * 2 - 21) // 2 + 1
        input_size = (input_size + 10 * 2 - 21) // 2 + 1

        self.rnn = nn.Sequential(
            RNNBlock(input_size * 96, fc_hidden, batch_first=True, dropout=dropout,batch_norm=False),
            *[
                RNNBlock(fc_hidden, fc_hidden, batch_first=True, dropout=dropout, batch_norm=True)
                for _ in range(rnn_layers - 1)
            ],
        )
        self.bn = nn.BatchNorm1d(fc_hidden)
        self.linear = nn.Linear(fc_hidden, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        spectrogram = spectrogram.unsqueeze(1)  # add channel dim
        x, length = self.conv(spectrogram, spectrogram_length)  # B x C x F x T
        b, c, f, t = x.shape
        x = x.view(b, c * f, t).transpose(1, 2)  # B x T x C * F
        for rnn in self.rnn:
            x = rnn(x, length)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2).contiguous()
        logits = self.linear(x)  # B x T x n_tokens
        return {
            "log_probs": F.log_softmax(logits, dim=-1),
            "log_probs_length": length,
            "probs": F.softmax(logits, dim=-1),
            "probs_length": length,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info