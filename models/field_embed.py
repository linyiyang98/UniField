"""B₀ field-strength embedding module.

Converts a scalar field-strength label (integer class index) into a
continuous embedding vector used for FiLM conditioning throughout the
UniField decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FieldEmbedding(nn.Module):
    """Learnable embedding for discrete B₀ field-strength labels.

    Parameters
    ----------
    num_field_strengths:
        Number of supported B₀ classes (default: 4 → 0.35T, 1.5T, 3T, 7T).
    embed_dim:
        Output embedding dimensionality.
    hidden_dim:
        Hidden layer size of the two-layer projection MLP.
    """

    def __init__(
        self,
        num_field_strengths: int = 4,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.lookup = nn.Embedding(num_field_strengths, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, field_labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        field_labels:
            Integer tensor of shape (B,) containing class indices.

        Returns
        -------
        torch.Tensor
            Field embedding of shape (B, embed_dim).
        """
        emb = self.lookup(field_labels)       # (B, embed_dim)
        return self.mlp(emb)                  # (B, embed_dim)
