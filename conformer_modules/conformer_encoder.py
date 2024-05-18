import torch
from torch import nn

from conformer_modules.conformer_block import ConformerBlock
from conformer_modules.convolution_subsampling import ConvolutionSubsampling


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        dropout: int = 0.1,
        feedforward_expansion_factor: int = 4,
        num_heads: int = 8,
        num_blocks=4,
        subsampling_stride: int = 2,
        subsampling_kernel_size: int = 4,
        pos_embed_max_len=1024,
    ) -> torch.tensor:
        super(ConformerEncoder, self).__init__()

        self.convsubsampling = ConvolutionSubsampling(
            in_channels=subsampling_stride * in_channels,
            stride=subsampling_stride,
            kernel_size=subsampling_kernel_size,
        )
        self.linear = nn.Linear(in_channels, in_channels)  # any changes here?
        self.dropout = nn.Dropout(dropout)
        self.conformer_blocks = nn.Sequential(
            *[
                ConformerBlock(
                    in_channels=in_channels,
                    dropout=dropout,
                    feedforward_expansion_factor=feedforward_expansion_factor,
                    num_heads=num_heads,
                    pos_embed_max_len=pos_embed_max_len,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        # x = self.convsubsampling(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.conformer_blocks(x)

        return x
