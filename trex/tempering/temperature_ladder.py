"""
Temperature ladder and swap schedule generation for Parallel Tempering.

This module provides:
- generate_temperature_ladder: Generate inverse temperature (beta) schedules
- get_swap_pairs: Get non-reversible swap pairs for a given timestep
"""

import torch
from typing import List, Tuple, Literal, Optional


def generate_temperature_ladder(
    num_temperatures: int,
    schedule: Literal["linear", "geometric", "quadratic"] = "linear",
    min_beta: float = 0.0,
    max_beta: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a temperature ladder (inverse temperature schedule).
    
    The temperature ladder defines the beta values for parallel tempering,
    where beta=0 is "hot" (explores freely) and beta=1 is "cold" (focused on high-reward).
    
    Args:
        num_temperatures: Number of temperature levels (K)
        schedule: How to space temperatures ("linear", "geometric", "quadratic")
        min_beta: Minimum inverse temperature (hottest)
        max_beta: Maximum inverse temperature (coldest)
        device: Device for the output tensor
        
    Returns:
        Tensor of beta values, shape (num_temperatures,), monotonically increasing
    """
    if num_temperatures < 1:
        raise ValueError("num_temperatures must be >= 1")
    
    if num_temperatures == 1:
        return torch.tensor([max_beta], device=device)
    
    # Generate unit interval values based on schedule
    t = torch.linspace(0, 1, num_temperatures, device=device)
    
    if schedule == "linear":
        # Linear spacing: betas are evenly spaced
        unit_values = t
        
    elif schedule == "geometric":
        # Geometric spacing: more temperatures at lower betas
        # Using log-space interpolation
        # Avoid log(0) by using a small epsilon
        eps = 1e-6
        log_min = torch.log(torch.tensor(eps, device=device))
        log_max = torch.log(torch.tensor(1.0, device=device))
        log_values = log_min + (log_max - log_min) * t
        unit_values = torch.exp(log_values)
        
    elif schedule == "quadratic":
        # Quadratic spacing: more temperatures at lower betas
        unit_values = t ** 2
        
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'linear', 'geometric', or 'quadratic'")
    
    # Scale to [min_beta, max_beta]
    betas = min_beta + (max_beta - min_beta) * unit_values
    
    return betas


def get_swap_pairs(
    timestep: int,
    num_temperatures: int,
) -> List[Tuple[int, int]]:
    """
    Get swap pairs for non-reversible parallel tempering.
    
    Uses an alternating schedule where:
    - Odd timesteps: swap pairs (0,1), (2,3), (4,5), ...
    - Even timesteps: swap pairs (1,2), (3,4), (5,6), ...
    
    This ensures all adjacent pairs are attempted over 2 timesteps,
    while avoiding overlapping swaps (no temperature appears twice per timestep).
    
    The non-reversible schedule improves mixing compared to random pair selection.
    
    Args:
        timestep: Current timestep (1-indexed, odd or even)
        num_temperatures: Total number of temperature levels (K)
        
    Returns:
        List of (i, j) pairs to attempt swapping, 0-indexed
    """
    if num_temperatures < 2:
        return []
    
    pairs = []
    
    # Determine starting index based on odd/even timestep
    if timestep % 2 == 1:
        # Odd timestep: (0,1), (2,3), (4,5), ...
        start = 0
    else:
        # Even timestep: (1,2), (3,4), (5,6), ...
        start = 1
    
    # Generate non-overlapping pairs
    i = start
    while i + 1 < num_temperatures:
        pairs.append((i, i + 1))
        i += 2
    
    return pairs
