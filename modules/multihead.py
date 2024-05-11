import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.prenorm = nn.LayerNorm(in_channels)
        self.multihead = nn.MultiheadAttention(in_channels, num_heads=num_heads)
        self.pos_embedding = RelativePositionalEmbedding(in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_forward = self.prenorm(x)
        x_forward = self.pos_embedding(x_forward)
        x_forward = self.multihead(x_forward, x_forward, x_forward)[0]
        x_forward = self.dropout(x_forward)
        return x + x_forward


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, input_dim, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(max_len, input_dim)

    def forward(self, x):
        # Generate relative positional embeddings
        _, seq_len = x.size(0), x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = self.embedding(pos)
        return pos
