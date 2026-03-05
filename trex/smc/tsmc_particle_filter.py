"""Archived TSMC particle filter kept only for backwards-compatible imports."""

from typing import Any

_ARCHIVED_MESSAGE = (
    "trex.smc.tsmc_particle_filter is archived after the March 2026 pivot away from twisted/transport SMC. "
    "Use the active multi-chain SMC paths instead; see docs/archive/2026-feb/ for historical reference."
)


class TSMCLLMParticleFilter:
    """Archived API shim."""

    def __init__(
        self,
        config: Any,
        generator: Any,
        twist_scorer: Any,
        reward_model: Any = None,
        answer_extractor: Any = None,
    ):
        raise RuntimeError(_ARCHIVED_MESSAGE)


__all__ = ["TSMCLLMParticleFilter"]
