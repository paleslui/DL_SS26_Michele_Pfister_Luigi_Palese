"""1D CNN over chromosome-ordered gene expression.

Architectural rationale
-----------------------
Genes ordered by chromosomal position carry local structure: nearby genes are
often co-expressed (same chromatin domain, shared regulatory elements, copy-
number variants affecting blocks). A 1D convolution over this ordered axis
can pick up such local genomic patterns.

Input shape : (batch, 1, n_genes)        - single channel = expression value
Architecture: N conv blocks → adaptive pool → MLP head → 1 logit
Each conv block : Conv1d → BatchNorm → ReLU → (Max|Avg)Pool → Dropout
The last block uses AdaptiveAvgPool1d(1) to collapse any remaining length.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_genes: int,
        # New flexible interface
        base_channels: int = 32,
        n_conv_blocks: int = 3,
        kernel_size: int = 9,
        pool_size: int = 4,
        pool_type: str = "max",
        dropout_conv: float = 0.3,
        dropout_head: float = 0.5,
        dense_dim: int = 64,
        # Backward-compat: explicit channels tuple overrides base_channels/n_conv_blocks
        channels: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()

        # Resolve channel sequence
        if channels is not None:
            ch_seq = list(channels)
            n_blocks = len(ch_seq)
        else:
            n_blocks = n_conv_blocks
            ch_seq = [base_channels * (2 ** i) for i in range(n_blocks)]

        # Pool layer factory
        if pool_type == "max":
            Pool = nn.MaxPool1d
        elif pool_type == "avg":
            Pool = nn.AvgPool1d
        else:
            raise ValueError(f"Unknown pool_type: {pool_type!r}")

        pad = kernel_size // 2

        # Build N blocks: first N-1 use Pool(pool_size), last uses AdaptiveAvgPool1d(1)
        blocks = []
        in_ch = 1
        for i, out_ch in enumerate(ch_seq):
            is_last = (i == n_blocks - 1)
            block_layers = [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if is_last:
                block_layers.append(nn.AdaptiveAvgPool1d(1))
            else:
                block_layers.append(Pool(pool_size))
                block_layers.append(nn.Dropout(dropout_conv))
            blocks.append(nn.Sequential(*block_layers))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        # Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch_seq[-1], dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_genes) → add channel dim → (batch, 1, n_genes)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)  # (batch,) logits
