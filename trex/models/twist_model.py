"""Archived twist-model module kept only for backwards-compatible imports."""

from dataclasses import dataclass
from typing import Any, Literal, Optional

TwistSpace = Literal["log_prob", "prob"]

_ARCHIVED_MESSAGE = (
    "trex.models.twist_model is archived after the March 2026 pivot away from twisted/value-head transport code. "
    "Do not use TwistModel in new runs; see docs/archive/2026-feb/ for historical reference."
)


@dataclass
class TwistModelConfig:
    """Archived API shim."""

    model_name_or_path: str = ""
    value_head_type: str = "mlp"
    twist_space: TwistSpace = "log_prob"
    freeze_base_model: bool = True
    share_base_with_generator: bool = False
    max_length: Optional[int] = None
    epsilon: float = 1e-8
    log_value_min: float = -1e6

    def __post_init__(self) -> None:
        raise RuntimeError(_ARCHIVED_MESSAGE)


class TwistModel:
    """Archived API shim."""

    def __init__(self, *args: Any, **kwargs: Any):
        raise RuntimeError(_ARCHIVED_MESSAGE)


__all__ = ["TwistSpace", "TwistModelConfig", "TwistModel"]
