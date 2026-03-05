"""Archival tests for obsolete value-head entry points."""

import pytest

from trex.models.value_head import (
    AttentionPooledValueHead,
    LinearValueHead,
    MLPValueHead,
    ValueHead,
)


@pytest.mark.parametrize(
    "cls",
    [ValueHead, LinearValueHead, MLPValueHead, AttentionPooledValueHead],
)
def test_value_head_classes_raise_archival_error(cls):
    """Obsolete value-head classes should fail fast after the March 2026 pivot."""
    with pytest.raises(RuntimeError, match="archived"):
        cls(hidden_dim=8)
