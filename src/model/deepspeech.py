import torch
from torch import nn


class NormBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, is_batch_norm):
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(
            input_size, hidden_size, bias=False, batch_first=True, bidirectional=True
        )
        self.is_batch_norm = is_batch_norm
        if self.is_batch_norm:
            self.bn = nn.BatchNorm1d(input_size)

    def forward(self, input):
        if self.is_batch_norm:
            batch_size, length, dim = input.shape
            input = self.bn(input.view(-1, dim))
            input = input.view(batch_size, length, dim)
        input, _ = self.gru(input)
        batch_size, length, dim = input.shape
        input = input.view(batch_size, length, 2, -1).sum(dim=2)
        input = input.view(batch_size, length, -1)
        return input


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_feats,
        n_class,
        n_layers=3,
        hidden_size=1024,
        n_channels=[32, 32],
        kernel_size=[(41, 11), (21, 11)],
        stride=[(2, 2), (2, 1)],
        padding=[(20, 5), (10, 5)],
        **batch,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.n_channels[0],
                kernel_size=self.kernel_size[0],
                stride=self.stride[0],
                padding=self.padding[0],
            ),
            nn.BatchNorm2d(n_channels[0]),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(
                in_channels=self.n_channels[0],
                out_channels=self.n_channels[1],
                kernel_size=self.kernel_size[1],
                stride=self.stride[1],
                padding=self.padding[1],
            ),
            nn.BatchNorm2d(self.n_channels[1]),
            nn.Hardtanh(0, 20, inplace=True),
        )

        self.input_size = n_feats * 8

        self.first_gru = NormBiGRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            is_batch_norm=False,
        )
        layers = []

        for i in range(n_layers - 1):
            layers.append(
                NormBiGRU(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    is_batch_norm=False,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_features=hidden_size, out_features=n_class)

    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.unsqueeze(1)
        spectrogram = self.conv(spectrogram).permute(0, 3, 1, 2)
        spectrogram = spectrogram.view(spectrogram.shape[0], spectrogram.shape[1], -1)
        spectrogram = self.first_gru(spectrogram)
        spectrogram = self.layers(spectrogram)
        spectrogram = self.linear(spectrogram)
        log_probs = nn.functional.log_softmax(spectrogram, dim=-1)
        log_probs_length = self.transform_input_lengths(batch["spectrogram_length"])
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        new_size = torch.tensor(
            input_lengths + 2 * self.padding[1][0] - (self.kernel_size[1][0] - 1) - 1,
            dtype=torch.float64,
        )
        return torch.floor(new_size / self.stride[1][0] + 1).int()

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
