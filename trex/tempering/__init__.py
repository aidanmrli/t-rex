"""
Tempering module for T-REX.

This module provides parallel tempering functionality:
- temperature_ladder: Temperature schedule generation and swap scheduling
- exchange: Replica exchange Metropolis-Hastings steps
"""

from trex.tempering.temperature_ladder import (
    generate_temperature_ladder,
    get_swap_pairs,
)
from trex.tempering.exchange import (
    compute_acceptance_ratio,
    metropolis_hastings_accept,
    swap_replicas,
)

__all__ = [
    # Temperature Ladder
    "generate_temperature_ladder",
    "get_swap_pairs",
    # Exchange
    "compute_acceptance_ratio",
    "metropolis_hastings_accept",
    "swap_replicas",
]
