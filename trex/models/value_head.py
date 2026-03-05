"""Archived value-head module kept only for backwards-compatible imports."""

from typing import Any

_ARCHIVED_MESSAGE = (
    "trex.models.value_head is archived after the March 2026 pivot away from twisted/value-head transport code. "
    "Do not use value-head scoring in new runs; see docs/archive/2026-feb/ for historical reference."
)


class ValueHead:
    """Archived API shim."""

    per_token: bool = True

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError(_ARCHIVED_MESSAGE)


class LinearValueHead(ValueHead):
    """Archived API shim."""


class MLPValueHead(ValueHead):
    """Archived API shim."""


class AttentionPooledValueHead(ValueHead):
    """Archived API shim."""


__all__ = [
    "ValueHead",
    "LinearValueHead",
    "MLPValueHead",
    "AttentionPooledValueHead",
]
