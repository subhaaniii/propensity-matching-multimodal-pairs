from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DualEncoder(nn.Module):
    """
    Two-tower encoder for synthetic modality A and modality B features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.encoder_a = ProjectionMLP(input_dim, hidden_dim, embed_dim, dropout)
        self.encoder_b = ProjectionMLP(input_dim, hidden_dim, embed_dim, dropout)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder_a(x_a), self.encoder_b(x_b)

    def encode_a(self, x_a: torch.Tensor) -> torch.Tensor:
        return self.encoder_a(x_a)

    def encode_b(self, x_b: torch.Tensor) -> torch.Tensor:
        return self.encoder_b(x_b)