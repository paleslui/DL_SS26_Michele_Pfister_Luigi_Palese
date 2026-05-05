"""1D CNN over chromosome-ordered gene expression.

Architectural rationale
-----------------------
Genes ordered by chromosomal position carry local structure: nearby genes are
often co-expressed (same chromatin domain, shared regulatory elements, copy-
number variants affecting blocks). A 1D convolution over this ordered axis
can pick up such local genomic patterns.

Input shape : (batch, 1, n_genes)        — single channel = expression value
Architecture: 3 conv blocks → adaptive pool → MLP head → 1 logit
Each conv block : Conv1d → BatchNorm → ReLU → MaxPool → Dropout
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(
        self,
        n_genes: int,
        channels: tuple[int, int, int] = (32, 64, 128),
        kernel_size: int = 9,
        pool_size: int = 4,
        dropout_conv: float = 0.3,
        dropout_head: float = 0.5,
        dense_dim: int = 64,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        pad = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_conv),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout_conv),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # collapses any remaining length to 1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(dense_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_genes) → add channel dim → (batch, 1, n_genes)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.head(x).squeeze(-1)  # (batch,) logits
