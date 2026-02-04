"""
SMC (Sequential Monte Carlo) module for T-REX.

This module provides core SMC functionality including:
- resampling: Resampling algorithms (multinomial, systematic, stratified)
- particle_filter: ParticleFilter class for managing particles and weights
- twisted_smc: Twisted SMC with value-function guided sampling
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
from trex.smc.twisted_smc import (
    compute_twisted_weights,
    TwistedSMCConfig,
    TwistedSMC,
)
from trex.smc.llm_particle_filter import LLMParticleFilter
from trex.smc.tsmc_particle_filter import TSMCLLMParticleFilter

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
    # Twisted SMC
    "compute_twisted_weights",
    "TwistedSMCConfig",
    "TwistedSMC",
    # LLM Particle Filter
    "LLMParticleFilter",
    "TSMCLLMParticleFilter",
]

