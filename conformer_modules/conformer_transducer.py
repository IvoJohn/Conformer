import torch
from torch import nn

from .conformer_encoder import ConformerEncoder


class ConformerTransducer(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        encoder_dim: int = 144,
        vocab_size: int = 1000,
        dropout: int = 0.1,
        feedforward_expansion_factor: int = 4,
        num_heads: int = 8,
        num_blocks: int = 16,
        subsampling_stride: int = 10,
        subsampling_kernel_size: int = 32,
        pos_embed_max_len: int = 1024,
        decoder_layers: int = 1,  # lstm layers
        decoder_dim: int = 320,  # lstm hidden size
    ) -> torch.tensor:
        super(ConformerTransducer, self).__init__()

        self.conformer_encoder = ConformerEncoder(
            encoder_dim=encoder_dim,
            in_channels=in_channels,
            feedforward_expansion_factor=feedforward_expansion_factor,
            num_heads=num_heads,
            num_blocks=num_blocks,
            subsampling_stride=subsampling_stride,
            subsampling_kernel_size=subsampling_kernel_size,
            pos_embed_max_len=pos_embed_max_len,
            dropout=dropout,
        )

        self.decoder = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=decoder_dim,
            num_layers=decoder_layers,
            batch_first=True,
        )

        self.linear_output = nn.Linear(decoder_dim, vocab_size)

    def forward(self, x):
        x = self.conformer_encoder(x)
        x, _ = self.decoder(x)
        x = self.linear_output(x)

        return x
