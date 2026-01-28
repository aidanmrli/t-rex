"""
Twisted Sequential Monte Carlo (TSMC) implementation using PyTorch.

Twisted SMC uses a learned value function (twist) to guide sampling
towards high-reward regions. The twist function modifies importance weights
to prefer particles that are likely to lead to good outcomes.

This module provides:
- compute_twisted_weights: Compute twisted importance weight ratios
- TwistedSMCConfig: Configuration for Twisted SMC
- TwistedSMC: Main class extending ParticleFilter with twisted proposal
"""

import torch
from dataclasses import dataclass
from typing import Optional, Callable, List

from trex.smc.particle_filter import SMCConfig, ParticleFilter
from trex.smc.resampling import normalize_weights


@dataclass
class TwistedSMCConfig(SMCConfig):
    """
    Configuration for Twisted Sequential Monte Carlo.
    
    Extends SMCConfig with twist-specific parameters.
    """
    
    # Whether to use the twist function for importance weighting
    use_twist: bool = True
    
    # Epsilon for numerical stability (avoid division by zero)
    epsilon: float = 1e-8
    
    # Temperature for soft-max style weighting (higher = more greedy)
    temperature: float = 1.0


def compute_twisted_weights(
    values_t: torch.Tensor,
    values_t_minus_1: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute twisted importance weight ratios.
    
    The twisted weight for particle i at time t is:
        w_t^i = V_t(x_t^i) / V_{t-1}(x_{t-1}^i)
    
    Where V_t is the value function estimate at time t.
    This biases sampling towards particles with improving values.
    
    For log-space values (e.g., log-probabilities), this becomes:
        w_t^i = exp(V_t - V_{t-1})
    
    Args:
        values_t: Value estimates at current time, shape (n_particles,)
        values_t_minus_1: Value estimates at previous time, shape (n_particles,)
        epsilon: Small value for numerical stability
        
    Returns:
        Weight ratios tensor, shape (n_particles,)
    """
    # Handle the case where we're working in probability space (values in [0, 1])
    # vs log-space (values can be negative)
    
    # Check if values look like they're in log-space (can be negative)
    in_log_space = torch.any(values_t < 0) or torch.any(values_t_minus_1 < 0)
    
    if in_log_space:
        # Log-space: weight ratio = exp(V_t - V_{t-1})
        # This is numerically stable
        weight_ratios = torch.exp(values_t - values_t_minus_1)
    else:
        # Probability space: weight ratio = V_t / V_{t-1}
        # Add epsilon to denominator for numerical stability
        denominator = values_t_minus_1 + epsilon
        weight_ratios = values_t / denominator
    
    return weight_ratios


class TwistedSMC(ParticleFilter):
    """
    Twisted Sequential Monte Carlo sampler.
    
    Extends ParticleFilter with twist-based importance weighting:
    - Uses a value function to estimate future rewards
    - Updates particle weights based on value improvements
    - Supports adaptive resampling with twist-aware ESS
    """
    
    def __init__(self, config: TwistedSMCConfig):
        """
        Initialize Twisted SMC.
        
        Args:
            config: TwistedSMCConfig with TSMC parameters
        """
        super().__init__(config)
        self.config: TwistedSMCConfig = config
        
        # Value function (can be set externally)
        self.value_function: Optional[Callable[[List[str]], torch.Tensor]] = None
        
        # Previous value estimates for computing weight ratios
        self._previous_values: Optional[torch.Tensor] = None
        
        # Current value estimates
        self._current_values: Optional[torch.Tensor] = None
    
    def set_value_function(self, value_fn: Callable[[List[str]], torch.Tensor]) -> None:
        """
        Set the value function used for twisted importance weights.
        
        Args:
            value_fn: Function that takes list of texts and returns value tensor
        """
        self.value_function = value_fn
    
    def compute_values(self) -> torch.Tensor:
        """
        Compute values for all particles using the value function.
        
        Returns:
            Value tensor, shape (n_particles,)
            
        Raises:
            ValueError: If value function not set
        """
        if self.value_function is None:
            raise ValueError("Value function not set. Call set_value_function() first.")
        
        texts = self.get_particle_texts()
        return self.value_function(texts)
    
    def get_previous_values(self) -> Optional[torch.Tensor]:
        """Get the previous value estimates."""
        return self._previous_values
    
    def set_current_values(self, values: torch.Tensor) -> None:
        """
        Set current values and move previous values to history.
        
        Args:
            values: Current value estimates
        """
        self._previous_values = self._current_values
        self._current_values = values.clone().detach()
        
        # Initialize previous values if this is the first set
        if self._previous_values is None:
            self._previous_values = values.clone().detach()
    
    def update_weights_with_twist(
        self,
        current_values: torch.Tensor,
        previous_values: torch.Tensor,
    ) -> None:
        """
        Update particle weights using twisted importance weights.
        
        Computes weight ratios based on value improvement and multiplies
        with current weights. Result is normalized.
        
        Args:
            current_values: Value estimates at current step
            previous_values: Value estimates at previous step
        """
        if not self.config.use_twist:
            return
        
        # Compute twisted weight ratios
        weight_ratios = compute_twisted_weights(
            current_values,
            previous_values,
            epsilon=self.config.epsilon,
        )
        
        # Multiply current weights by ratios
        current_weights = self.get_weights()
        new_weights = current_weights * weight_ratios
        
        # Normalize
        new_weights = normalize_weights(new_weights)
        self.set_weights(new_weights)
        
        # Store values for next step
        self._previous_values = previous_values.clone().detach()
        self._current_values = current_values.clone().detach()
    
    def step_with_twist(self) -> None:
        """
        Single step of Twisted SMC.
        
        1. Compute values for current particles
        2. Update weights with twist
        3. Resample if ESS is low
        """
        if self.value_function is None:
            raise ValueError("Value function not set.")
        
        # Get current values
        current_values = self.compute_values()
        
        # Get previous values (or initialize to current)
        if self._previous_values is None:
            self._previous_values = current_values.clone()
        
        previous_values = self._previous_values
        
        # Update weights with twist
        self.update_weights_with_twist(current_values, previous_values)
        
        # Adaptive resampling
        if self.should_resample():
            self.resample()
        
        # Store current values for next step
        self._current_values = current_values
        self._previous_values = current_values.clone()
