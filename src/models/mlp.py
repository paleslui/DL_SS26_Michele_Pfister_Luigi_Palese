"""Simple MLP for tabular feature classification.

Architecture: input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear (logit)

Used for:
  - Model 2 (50 pathway scores → MSI prediction)
  - Could be reused for any tabular feature set
"""
from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # raw logit; sigmoid applied via BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,) logits
