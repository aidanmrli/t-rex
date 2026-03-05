"""
SMC (Sequential Monte Carlo) module for T-REX.

This module provides core SMC functionality including:
- resampling: Resampling algorithms (multinomial, systematic, stratified)
- particle_filter: ParticleFilter class for managing particles and weights
- single_chain_smc: Stage 3 single-chain SMC core
- multi_chain_smc: Stage 4 multi-chain SMC core
"""

from trex.smc.resampling import (
    multinomial_resampling,
    systematic_resampling,
    stratified_resampling,
    normalize_weights,
    compute_ess,
)
from trex.smc.particle_filter import (
    SMCConfig,
    Particle,
    ParticleFilter,
)
from trex.smc.llm_particle_filter import LLMParticleFilter
from trex.smc.single_chain_smc import SingleChainSMC, StepResult
from trex.smc.multi_chain_smc import MultiChainSMC, MultiChainSMCConfig, MultiChainSMCResult

__all__ = [
    # Resampling
    "multinomial_resampling",
    "systematic_resampling",
    "stratified_resampling",
    "normalize_weights",
    "compute_ess",
    # Particle Filter
    "SMCConfig",
    "Particle",
    "ParticleFilter",
    # LLM Particle Filter
    "LLMParticleFilter",
    # Stage 3/4 Core
    "StepResult",
    "SingleChainSMC",
    "MultiChainSMCConfig",
    "MultiChainSMCResult",
    "MultiChainSMC",
]
