import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 128,
        dropout=0.1,
        expansion_factor: int = 4,
    ):
        super(FeedForward, self).__init__()

        self.prenorm = nn.LayerNorm([encoder_dim])
        self.linear = nn.Linear(encoder_dim, expansion_factor * encoder_dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion_factor * encoder_dim, encoder_dim)

    def forward(self, x):
        x_forward = self.prenorm(x)
        x_forward = self.linear(x_forward)
        x_forward = self.swish(x_forward)
        x_forward = self.dropout(x_forward)
        x_forward = self.linear2(x_forward)
        return x + x_forward
