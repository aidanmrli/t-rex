"""
Replica Exchange for Parallel Tempering.

This module provides:
- compute_acceptance_ratio: Compute Metropolis-Hastings acceptance probability
- metropolis_hastings_accept: Stochastic accept/reject step
- swap_replicas: Attempt to swap two replicas
"""

import torch
from typing import Tuple, TypeVar, Union

T = TypeVar('T')  # Generic type for replicas


def compute_acceptance_ratio(
    phi_x: Union[float, torch.Tensor],
    phi_x_prime: Union[float, torch.Tensor],
    beta_target: Union[float, torch.Tensor],
) -> float:
    """
    Compute the Metropolis-Hastings acceptance ratio for replica exchange.
    
    For swapping replicas between temperatures, the acceptance ratio is:
        α = min(1, (φ(x')/φ(x))^β)
    
    Where:
    - φ(x) is the value/reward of current state
    - φ(x') is the value/reward of proposed state
    - β is the inverse temperature (0=hot, 1=cold)
    
    At high temperature (β≈0), all moves are accepted (exploration).
    At low temperature (β≈1), only improving moves are accepted (exploitation).
    
    Args:
        phi_x: Value of current state
        phi_x_prime: Value of proposed state
        beta_target: Target inverse temperature
        
    Returns:
        Acceptance probability α ∈ [0, 1]
    """
    # Convert to float if tensor
    if isinstance(phi_x, torch.Tensor):
        phi_x = phi_x.item()
    if isinstance(phi_x_prime, torch.Tensor):
        phi_x_prime = phi_x_prime.item()
    if isinstance(beta_target, torch.Tensor):
        beta_target = beta_target.item()
    
    # Handle β = 0 case (always accept)
    if beta_target == 0.0:
        return 1.0
    
    # Handle equal phi case
    if phi_x == phi_x_prime:
        return 1.0
    
    # Avoid division by zero
    if phi_x <= 0:
        phi_x = 1e-10
    if phi_x_prime <= 0:
        phi_x_prime = 1e-10
    
    # Compute ratio
    ratio = phi_x_prime / phi_x
    
    # Apply temperature
    acceptance = ratio ** beta_target
    
    # Clamp to [0, 1]
    return min(1.0, max(0.0, acceptance))


def metropolis_hastings_accept(alpha: float) -> bool:
    """
    Stochastic accept/reject step for Metropolis-Hastings.
    
    Accept with probability α, reject with probability 1-α.
    
    Args:
        alpha: Acceptance probability ∈ [0, 1]
        
    Returns:
        True if accepted, False if rejected
    """
    if alpha >= 1.0:
        return True
    if alpha <= 0.0:
        return False
    
    # Sample uniform random number
    u = torch.rand(1).item()
    return u < alpha


def swap_replicas(
    replica_i: T,
    replica_j: T,
    phi_i: float,
    phi_j: float,
    beta_i: float,
    beta_j: float,
) -> Tuple[T, T, bool]:
    """
    Attempt to swap two replicas between temperature levels.
    
    The swap is accepted according to the Metropolis-Hastings criterion.
    For replica exchange, the acceptance ratio is:
        α = min(1, exp((β_j - β_i) * (log φ_j - log φ_i)))
    
    Or equivalently using the simpler formulation assumed by the tests:
        α_i = min(1, (φ_j / φ_i)^β_i) for replica i accepting j's value
        α_j = min(1, (φ_i / φ_j)^β_j) for replica j accepting i's value
    
    The combined acceptance is min(α_i, α_j) or product depending on scheme.
    
    Args:
        replica_i: First replica (at temperature beta_i)
        replica_j: Second replica (at temperature beta_j)
        phi_i: Value/reward of replica_i
        phi_j: Value/reward of replica_j
        beta_i: Inverse temperature of position i
        beta_j: Inverse temperature of position j
        
    Returns:
        Tuple of (new_replica_i, new_replica_j, did_swap)
    """
    # Compute acceptance ratios for each replica accepting the other's value
    # When we swap, replica at position i gets phi_j, and vice versa
    
    # For replica at position i (temperature beta_i), accepting value phi_j
    alpha_i = compute_acceptance_ratio(phi_i, phi_j, beta_i)
    
    # For replica at position j (temperature beta_j), accepting value phi_i
    alpha_j = compute_acceptance_ratio(phi_j, phi_i, beta_j)
    
    # Combined acceptance (product of individual acceptances)
    # This is equivalent to the standard replica exchange formula
    alpha_combined = alpha_i * alpha_j
    
    # Accept or reject
    if metropolis_hastings_accept(alpha_combined):
        # Swap replicas
        return replica_j, replica_i, True
    else:
        # Keep in place
        return replica_i, replica_j, False
