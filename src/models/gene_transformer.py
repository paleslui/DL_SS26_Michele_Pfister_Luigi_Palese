"""Gene-attention transformer for tabular gene expression.

Each gene becomes a token. The token's representation is the sum of:
  - a LEARNED embedding for the gene's identity (size n_genes × d_model)
  - a per-sample expression value, projected to d_model
The sequence of n_genes tokens passes through a small Transformer encoder.
A [CLS]-style learned pooling token then feeds the classification head.

This treats genes as an UNORDERED set (no positional encoding) — the model
learns gene-to-gene relationships via self-attention. Attention weights from
the [CLS] token are interpretable: which genes did the model attend to when
making this prediction?
"""
from __future__ import annotations

import torch
from torch import nn


class GeneTransformer(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Learned gene-identity embeddings (one vector per gene)
        self.gene_embedding = nn.Embedding(n_genes, d_model)

        # Project the scalar expression value into d_model space
        self.expression_proj = nn.Linear(1, d_model)

        # Learned [CLS] token for sequence-level classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # Cached register for gene index tensor
        self.register_buffer(
            "gene_idx",
            torch.arange(n_genes).unsqueeze(0),  # (1, n_genes)
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_genes) of expression values
        batch = x.size(0)

        # Gene-identity embedding broadcast across the batch
        gene_emb = self.gene_embedding(self.gene_idx).expand(batch, -1, -1)
        # Expression projection: (batch, n_genes, 1) → (batch, n_genes, d_model)
        expr_emb = self.expression_proj(x.unsqueeze(-1))

        tokens = gene_emb + expr_emb  # (batch, n_genes, d_model)

        # Prepend [CLS]
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Encode and use the [CLS] hidden state
        out = self.encoder(tokens)
        cls_out = out[:, 0]
        return self.head(cls_out).squeeze(-1)
