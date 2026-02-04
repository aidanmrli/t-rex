"""
Unit tests for value head architectures.
"""

import torch

from trex.models.value_head import (
    AttentionPooledValueHead,
    LinearValueHead,
    MLPValueHead,
)


def test_linear_value_head_shape():
    """Linear head outputs per-token logits of shape [B, T, 1]."""
    head = LinearValueHead(hidden_dim=8)
    hidden = torch.randn(2, 3, 8)
    out = head(hidden)
    assert out.shape == (2, 3, 1)


def test_mlp_value_head_shape():
    """MLP head outputs per-token logits of shape [B, T, 1]."""
    head = MLPValueHead(hidden_dim=8)
    hidden = torch.randn(2, 4, 8)
    out = head(hidden)
    assert out.shape == (2, 4, 1)


def test_attention_pooled_value_head_shape():
    """Attention-pooled head outputs sequence-level logits of shape [B, 1]."""
    head = AttentionPooledValueHead(hidden_dim=8)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    out = head(hidden, attention_mask=mask)
    assert out.shape == (2, 1)
