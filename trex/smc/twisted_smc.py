"""Archived twisted-SMC module kept only for backwards-compatible imports."""

from typing import Any

_ARCHIVED_MESSAGE = (
    "trex.smc.twisted_smc is archived after the March 2026 pivot away from twisted/transport SMC. "
    "Do not use TwistedSMC in new runs; see docs/archive/2026-feb/ for historical reference."
)


def _raise_archived() -> None:
    raise RuntimeError(_ARCHIVED_MESSAGE)


def compute_twisted_weights(*args: Any, **kwargs: Any):
    """Archived API shim."""
    _raise_archived()


class TwistedSMCConfig:
    """Archived API shim."""

    def __init__(self, *args: Any, **kwargs: Any):
        _raise_archived()


class TwistedSMC:
    """Archived API shim."""

    def __init__(self, config: Any):
        _raise_archived()


__all__ = ["compute_twisted_weights", "TwistedSMCConfig", "TwistedSMC"]
