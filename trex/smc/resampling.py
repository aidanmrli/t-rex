"""
Resampling algorithms for Sequential Monte Carlo (SMC) using PyTorch.

This module provides resampling algorithms and weight utilities:
- multinomial_resampling: Standard multinomial (categorical) resampling
- systematic_resampling: Low-variance systematic resampling
- stratified_resampling: Stratified resampling ensuring coverage
- normalize_weights: Weight normalization utility
- compute_ess: Effective Sample Size (ESS) calculation
"""

import torch
from typing import Optional


def normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Normalize weights to sum to 1.0.
    
    Args:
        weights: Unnormalized weight tensor, shape (n_particles,)
        
    Returns:
        Normalized weight tensor summing to 1.0
        
    Raises:
        ValueError: If all weights are zero or any weight is negative
    """
    if torch.any(weights < 0):
        raise ValueError("Negative weights are not allowed in SMC resampling.")
    
    weight_sum = weights.sum()
    if weight_sum == 0:
        raise ValueError("All weights are zero. Cannot normalize.")
    
    return weights / weight_sum


def compute_ess(weights: torch.Tensor) -> float:
    """
    Compute the Effective Sample Size (ESS).
    
    ESS = 1 / sum(w_i^2) for normalized weights.
    ESS indicates the "effective" number of particles contributing to the estimate.
    - ESS = n means uniform weights (all particles equally useful)
    - ESS = 1 means degenerate (one particle dominates)
    
    Args:
        weights: Normalized weight tensor, shape (n_particles,)
        
    Returns:
        Effective sample size (float between 1 and n_particles)
    """
    # Ensure weights are normalized
    w = weights / weights.sum()
    ess = 1.0 / (w ** 2).sum()
    return ess.item()


def multinomial_resampling(
    weights: torch.Tensor, 
    n_particles: int
) -> torch.LongTensor:
    """
    Multinomial (categorical) resampling.
    
    Draw n_particles indices with replacement, where P(index=i) ∝ weights[i].
    This is the simplest resampling algorithm but has high variance.
    
    Args:
        weights: Weight tensor, shape (n_particles_in,). Need not be normalized.
        n_particles: Number of particles to sample
        
    Returns:
        LongTensor of indices, shape (n_particles,), values in [0, len(weights)-1]
    """
    # Normalize weights for multinomial sampling
    probs = weights / weights.sum()
    
    # torch.multinomial samples indices with replacement based on probabilities
    indices = torch.multinomial(probs, n_particles, replacement=True)
    
    return indices.long()


def systematic_resampling(
    weights: torch.Tensor,
    u: Optional[float] = None
) -> torch.LongTensor:
    """
    Systematic resampling (low-variance resampling).
    
    Uses a single random offset u ∈ [0, 1/n) and equally spaced points
    to select particles. This reduces variance compared to multinomial.
    
    Args:
        weights: Weight tensor, shape (n_particles,). Need not be normalized.
        u: Starting offset in [0, 1/n). If None, drawn uniformly.
        
    Returns:
        LongTensor of indices, shape (n_particles,)
    """
    n = len(weights)
    device = weights.device
    
    # Normalize weights
    w = weights / weights.sum()
    
    # Compute cumulative sum
    cumsum = torch.cumsum(w, dim=0)
    
    # Generate uniform offset if not provided
    if u is None:
        u = torch.rand(1, device=device).item() / n
    
    # Equally spaced points starting at u
    positions = torch.arange(n, device=device, dtype=weights.dtype) / n + u
    
    # Find indices using searchsorted
    indices = torch.searchsorted(cumsum, positions)
    
    # Clamp to valid range (handles numerical edge cases)
    indices = torch.clamp(indices, 0, n - 1)
    
    return indices.long()


def stratified_resampling(weights: torch.Tensor) -> torch.LongTensor:
    """
    Stratified resampling.
    
    Divides [0, 1] into n equal strata and draws one random point per stratum.
    This ensures each stratum contributes exactly one particle.
    
    Args:
        weights: Weight tensor, shape (n_particles,). Need not be normalized.
        
    Returns:
        LongTensor of indices, shape (n_particles,)
    """
    n = len(weights)
    device = weights.device
    
    # Normalize weights
    w = weights / weights.sum()
    
    # Compute cumulative sum
    cumsum = torch.cumsum(w, dim=0)
    
    # Generate one random point per stratum: u_i ~ Uniform(i/n, (i+1)/n)
    strata = torch.arange(n, device=device, dtype=weights.dtype)
    random_offsets = torch.rand(n, device=device, dtype=weights.dtype)
    positions = (strata + random_offsets) / n
    
    # Find indices using searchsorted
    indices = torch.searchsorted(cumsum, positions)
    
    # Clamp to valid range
    indices = torch.clamp(indices, 0, n - 1)
    
    return indices.long()
