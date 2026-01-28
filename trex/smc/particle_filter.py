"""
Particle Filter implementation for Sequential Monte Carlo (SMC) using PyTorch.

This module provides:
- SMCConfig: Configuration dataclass for SMC parameters
- Particle: Dataclass representing a single particle
- ParticleFilter: Main class for managing particles and weights
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union
from copy import deepcopy

from trex.smc.resampling import (
    multinomial_resampling,
    systematic_resampling,
    stratified_resampling,
    normalize_weights,
    compute_ess,
)


@dataclass
class SMCConfig:
    """Configuration for Sequential Monte Carlo parameters."""
    
    # Number of particles
    n_particles: int = 16
    
    # ESS threshold for adaptive resampling (fraction of n_particles)
    # If ESS < ess_threshold * n_particles, trigger resampling
    ess_threshold: float = 0.5
    
    # Resampling algorithm to use
    resampling_method: Literal["multinomial", "systematic", "stratified"] = "systematic"
    
    # Device for tensor operations
    device: str = "cpu"
    
    # Random seed for reproducibility (optional)
    seed: Optional[int] = None


@dataclass
class Particle:
    """
    Represents a single particle in the SMC algorithm.
    
    Attributes:
        text: The generated text (partial or complete)
        weight: The (unnormalized) weight of this particle
        log_weight: Optional log-weight for numerical stability
        metadata: Optional dictionary for additional particle data
    """
    text: str
    weight: float = 1.0
    log_weight: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class ParticleFilter:
    """
    Particle Filter for Sequential Monte Carlo sampling.
    
    Manages a set of particles with weights, supporting:
    - Initialization from a prompt
    - Weight updates and normalization
    - Resampling when ESS is low
    - ESS computation for adaptive resampling
    """
    
    def __init__(self, config: SMCConfig):
        """
        Initialize the particle filter.
        
        Args:
            config: SMCConfig with SMC parameters
        """
        self.config = config
        self.particles: List[Particle] = []
        self._weights: Optional[torch.Tensor] = None
        self._device = torch.device(config.device)
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
    
    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return len(self.particles)
    
    def initialize(self, prompt: str) -> None:
        """
        Initialize particles with the given prompt.
        
        Creates n_particles copies, each starting with the prompt.
        All weights are initialized to uniform (1/n).
        
        Args:
            prompt: Starting text for all particles
        """
        n = self.config.n_particles
        self.particles = [
            Particle(text=prompt, weight=1.0 / n)
            for _ in range(n)
        ]
        self._weights = torch.ones(n, device=self._device, dtype=torch.float32) / n
    
    def get_weights(self) -> torch.Tensor:
        """
        Get the current normalized weights as a tensor.
        
        Returns:
            Tensor of shape (n_particles,) summing to 1.0
        """
        if self._weights is None:
            raise ValueError("Particle filter not initialized. Call initialize() first.")
        return self._weights
    
    def set_weights(self, weights: Union[torch.Tensor, np.ndarray, List[float]]) -> None:
        """
        Set the weights from a tensor, numpy array, or list.
        
        Args:
            weights: New weights (need not be normalized, will be stored as-is)
        """
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).to(self._device)
        elif isinstance(weights, list):
            weights = torch.tensor(weights, device=self._device)
        elif isinstance(weights, torch.Tensor):
            weights = weights.to(self._device)
        
        if len(weights) != len(self.particles):
            raise ValueError(
                f"Weight count ({len(weights)}) doesn't match particle count ({len(self.particles)})"
            )
        
        self._weights = weights.float()
        
        # Also update individual particle weights for consistency
        for i, p in enumerate(self.particles):
            p.weight = weights[i].item()
    
    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        if self._weights is None:
            raise ValueError("No weights to normalize.")
        self._weights = normalize_weights(self._weights)
        
        # Update individual particle weights
        for i, p in enumerate(self.particles):
            p.weight = self._weights[i].item()
    
    def effective_sample_size(self) -> float:
        """
        Compute the Effective Sample Size (ESS).
        
        Returns:
            ESS value between 1 and n_particles
        """
        if self._weights is None:
            raise ValueError("No weights. Call initialize() first.")
        return compute_ess(self._weights)
    
    def should_resample(self) -> bool:
        """
        Check if resampling is needed based on ESS threshold.
        
        Returns:
            True if ESS < threshold * n_particles
        """
        ess = self.effective_sample_size()
        return ess < self.config.ess_threshold * self.config.n_particles
    
    def resample(self) -> None:
        """
        Resample particles based on their weights.
        
        Uses the configured resampling method. After resampling:
        - High-weight particles are duplicated
        - Low-weight particles are dropped
        - All weights are reset to uniform (1/n)
        """
        if self._weights is None:
            raise ValueError("No weights. Call initialize() first.")
        
        n = len(self.particles)
        
        # Get resampling indices based on configured method
        if self.config.resampling_method == "multinomial":
            indices = multinomial_resampling(self._weights, n)
        elif self.config.resampling_method == "systematic":
            indices = systematic_resampling(self._weights)
        elif self.config.resampling_method == "stratified":
            indices = stratified_resampling(self._weights)
        else:
            raise ValueError(f"Unknown resampling method: {self.config.resampling_method}")
        
        # Create new particle list by copying selected particles
        old_particles = self.particles
        self.particles = [
            deepcopy(old_particles[idx.item()])
            for idx in indices
        ]
        
        # Reset to uniform weights
        self._weights = torch.ones(n, device=self._device, dtype=torch.float32) / n
        for p in self.particles:
            p.weight = 1.0 / n
    
    def get_particle_texts(self) -> List[str]:
        """
        Get all particle texts.
        
        Returns:
            List of particle text strings
        """
        return [p.text for p in self.particles]
    
    def get_best_particle(self) -> Particle:
        """
        Get the particle with the highest weight.
        
        Returns:
            Particle with maximum weight
        """
        if not self.particles:
            raise ValueError("No particles initialized.")
        
        best_idx = self._weights.argmax().item()
        return self.particles[best_idx]
    
    def sample_particle(self) -> Particle:
        """
        Sample a particle proportional to its weight.
        
        Returns:
            A randomly sampled particle (weighted)
        """
        if not self.particles:
            raise ValueError("No particles initialized.")
        
        idx = torch.multinomial(self._weights, 1).item()
        return self.particles[idx]
