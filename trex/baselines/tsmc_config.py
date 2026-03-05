"""Archived configuration shim for the deprecated TSMC baseline."""

from dataclasses import dataclass
from typing import Any, Dict

from trex.baselines.smc_config import CheckpointManager, SMCSteeringConfig

_ARCHIVED_MESSAGE = (
    "trex.baselines.tsmc_config is archived after the March 2026 pivot away from twisted/transport SMC. "
    "TSMC baseline configs are disabled; see docs/archive/2026-feb/ for historical reference."
)


@dataclass
class TSMCConfig(SMCSteeringConfig):
    """Archived API shim."""

    def __post_init__(self) -> None:
        raise RuntimeError(_ARCHIVED_MESSAGE)

    def to_dict(self) -> Dict[str, Any]:
        raise RuntimeError(_ARCHIVED_MESSAGE)

    @classmethod
    def from_dict(cls, d):
        raise RuntimeError(_ARCHIVED_MESSAGE)

    def config_hash(self) -> str:
        raise RuntimeError(_ARCHIVED_MESSAGE)


__all__ = ["TSMCConfig", "CheckpointManager"]
