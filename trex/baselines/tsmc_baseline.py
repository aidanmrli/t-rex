"""Archived runner shim for the deprecated TSMC baseline."""

import argparse
from typing import Any, Dict

from trex.baselines.tsmc_config import TSMCConfig

_ARCHIVED_MESSAGE = (
    "trex.baselines.tsmc_baseline is archived after the March 2026 pivot away from twisted/transport SMC. "
    "TSMC baseline execution is disabled; see docs/archive/2026-feb/ for historical reference."
)


def _raise_archived() -> None:
    raise RuntimeError(_ARCHIVED_MESSAGE)


class TSMCBaseline:
    """Archived API shim."""

    def __init__(self, config: TSMCConfig):
        _raise_archived()

    def run(self) -> Dict[str, Any]:
        _raise_archived()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archived TSMC baseline runner.")
    parser.add_argument("--config", type=str, default=None)
    return parser


def load_config(args: argparse.Namespace) -> TSMCConfig:
    _raise_archived()


def main() -> None:
    _raise_archived()


if __name__ == "__main__":
    main()
