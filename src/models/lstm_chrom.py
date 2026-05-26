"""LSTM/GRU on chromosome-ordered gene expression.

Same input as the CNN (chromosome-ordered expression vector), so the comparison
is apples-to-apples: convolution vs recurrence on the same biological signal
(local genomic neighborhoods).

Input shape : (batch, n_genes)              - single channel, treated as sequence
Architecture: input projection → optional bidirectional LSTM/GRU → pooling → MLP head

Note on chunking: 27k genes is too long for a vanilla LSTM to process token-by-
token in reasonable time. Instead we 'pre-pool' the input by chunking adjacent
genes into windows, mean/max pooling each window, and feeding the pooled
sequence to the LSTM. This preserves chromosomal locality while making the
sequence length tractable (e.g. 27000 genes / chunk=50 = 540 timesteps).
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMChrom(nn.Module):
    def __init__(
        self,
        n_genes: int,
        chunk_size: int = 50,
        chunk_pool: str = "mean",       # 'mean' | 'max'
        rnn_type: str = "lstm",         # 'lstm' | 'gru'
        hidden_size: int = 64,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout_rnn: float = 0.2,
        dropout_head: float = 0.5,
        dense_dim: int = 64,
        sequence_pool: str = "last",    # 'last' | 'mean' | 'max'
    ) -> None:
        super().__init__()

        self.chunk_size = chunk_size
        self.chunk_pool = chunk_pool
        self.sequence_pool = sequence_pool
        self.bidirectional = bidirectional

        if rnn_type == "lstm":
            RNN = nn.LSTM
        elif rnn_type == "gru":
            RNN = nn.GRU
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type!r}")

        # Input dimension is 1 (we feed a single chunk-pooled scalar per timestep)
        # but we project it to a small embedding to give the RNN something to work with
        self.input_proj = nn.Linear(1, 16)

        self.rnn = RNN(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rnn if n_layers > 1 else 0.0,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Linear(out_size, dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(dense_dim, 1),
        )

    def _chunk_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool x of shape (batch, n_genes) into (batch, n_chunks).

        Trims any trailing genes that don't fit a full chunk (small loss)."""
        b, n = x.shape
        n_chunks = n // self.chunk_size
        x = x[:, : n_chunks * self.chunk_size]              # trim
        x = x.view(b, n_chunks, self.chunk_size)            # (b, n_chunks, chunk)
        if self.chunk_pool == "mean":
            x = x.mean(dim=-1)
        elif self.chunk_pool == "max":
            x = x.max(dim=-1).values
        else:
            raise ValueError(self.chunk_pool)
        return x  # (b, n_chunks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_genes)
        x = self._chunk_pool(x)                             # (b, n_chunks)
        x = self.input_proj(x.unsqueeze(-1))                # (b, n_chunks, 16)
        out, _ = self.rnn(x)                                 # (b, n_chunks, out_size)

        if self.sequence_pool == "last":
            x = out[:, -1, :]                               # final-step hidden
        elif self.sequence_pool == "mean":
            x = out.mean(dim=1)
        elif self.sequence_pool == "max":
            x = out.max(dim=1).values
        else:
            raise ValueError(self.sequence_pool)

        return self.head(x).squeeze(-1)
