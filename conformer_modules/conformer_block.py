import torch
from torch import nn

from .feedforward import FeedForward
from .multihead import MultiHeadSelfAttention
from .convolution import ConvolutionModule


class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 128,
        dropout: int = 0.1,
        feedforward_expansion_factor=4,
        num_heads=8,
        pos_embed_max_len=1024,
    ) -> torch.tensor:
        super(ConformerBlock, self).__init__()

        self.feedforward = FeedForward(
            encoder_dim=encoder_dim,
            dropout=dropout,
            expansion_factor=feedforward_expansion_factor,
        )
        self.multihead = MultiHeadSelfAttention(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            dropout=dropout,
            pos_embed_max_len=pos_embed_max_len,
        )
        self.conv = ConvolutionModule(encoder_dim=encoder_dim, dropout=dropout)
        self.feedforward2 = FeedForward(
            encoder_dim=encoder_dim,
            dropout=dropout,
            expansion_factor=feedforward_expansion_factor,
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)

    def forward(self, x):

        x = self.feedforward(x) + x
        x = self.multihead(x) + x
        x = self.conv(x) + x
        x = self.feedforward2(x) + x
        x = self.layer_norm(x)

        return x
