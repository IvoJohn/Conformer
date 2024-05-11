import torch
from torch import nn
from modules.feedforward import FeedForward
from modules.multihead import MultiHeadSelfAttention
from modules.convolution import ConvolutionModule


class ConformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        dropout: int = 0.1,
        feedforward_expansion_factor=4,
        num_heads=8,
    ) -> torch.tensor:
        super(ConformerBlock, self).__init__()

        self.feedforward = FeedForward(
            in_channels=in_channels,
            dropout=dropout,
            expansion_factor=feedforward_expansion_factor,
        )
        self.multihead = MultiHeadSelfAttention(
            in_channels=in_channels, num_heads=num_heads, dropout=dropout
        )
        self.conv = ConvolutionModule(in_channels=in_channels, dropout=dropout)
        self.feedforward2 = FeedForward(
            in_channels=in_channels,
            dropout=dropout,
            expansion_factor=feedforward_expansion_factor,
        )
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):

        x = self.feedforward(x) + x
        x = self.multihead(x) + x
        x = self.conv(x) + x
        x = self.feedforward2(x) + x
        x = self.layer_norm(x)

        return x
