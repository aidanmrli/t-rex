"""
Value head architectures for Twisted SMC.

All value heads output raw logits. Mapping to ψ or logψ is handled by the
twist model wrapper to avoid double activation.
"""

from typing import Optional

import torch
from torch import nn


class ValueHead(nn.Module):
    """Base class for value heads."""

    per_token: bool = True

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: Optional [batch_size, seq_len] mask (1=keep, 0=pad)
        Returns:
            logits: [batch_size, seq_len, 1] for per-token heads,
                    or [batch_size, 1] for pooled heads.
        """
        raise NotImplementedError


class LinearValueHead(ValueHead):
    """Simple linear projection to a single logit per token."""

    per_token: bool = True

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.linear(hidden_states)


class MLPValueHead(ValueHead):
    """Two-layer MLP with ReLU."""

    per_token: bool = True

    def __init__(self, hidden_dim: int, intermediate_dim: Optional[int] = None):
        super().__init__()
        intermediate_dim = intermediate_dim or max(1, hidden_dim // 4)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.layers(hidden_states)


class AttentionPooledValueHead(ValueHead):
    """Attention-pooled head for sequence-level value."""

    per_token: bool = False

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)
        self.scale = hidden_dim**0.5

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # hidden_states: [B, T, H]
        scores = (hidden_states * self.query).sum(dim=-1) / self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return self.out(pooled)
