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
    
    # Whether value function outputs are in log-space (can be negative)
    # If True: weight_ratio = exp(V_t - V_{t-1})
    # If False: weight_ratio = V_t / V_{t-1} (V must be in [0, inf))
    # Per HIGH_LEVEL_CONTEXT.md Section 2.4, twist uses sigmoid so default is False
    log_space: bool = False

    # Minimum log value when log_space=True (for numerical stability)
    log_value_min: float = -1e6


def compute_twisted_weights(
    values_t: torch.Tensor,
    values_t_minus_1: torch.Tensor,
    epsilon: float = 1e-8,
    log_space: bool = False,
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
        log_space: If True, values are in log-space and we compute exp(V_t - V_{t-1}).
                   If False, values are in probability space and we compute V_t / V_{t-1}.
                   Must be explicitly specified - no auto-detection.
        
    Returns:
        Weight ratios tensor, shape (n_particles,)
        
    Raises:
        ValueError: If log_space=False and values contain negative numbers
    """
    if log_space:
        # Log-space: weight ratio = exp(V_t - V_{t-1})
        # This is numerically stable
        weight_ratios = torch.exp(values_t - values_t_minus_1)
    else:
        # Probability space: weight ratio = V_t / V_{t-1}
        # Validate that values are non-negative
        if torch.any(values_t < 0) or torch.any(values_t_minus_1 < 0):
            raise ValueError(
                "Values contain negative numbers but log_space=False. "
                "Set log_space=True if values are in log-space."
            )
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
        values = self.value_function(texts)
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, device=self._device)
        return values.to(self._device).view(-1)
    
    def get_previous_values(self) -> Optional[torch.Tensor]:
        """Get the previous value estimates."""
        prev_values = self._get_prev_values_from_metadata()
        if prev_values is not None:
            return prev_values
        return self._previous_values
    
    def set_current_values(self, values: torch.Tensor) -> None:
        """
        Set current values and move previous values to history.
        
        Args:
            values: Current value estimates
        """
        values = values.detach().to(self._device)
        self._previous_values = values.clone().detach()
        self._current_values = values.clone().detach()
        self._set_prev_values_metadata(values)

    def _get_prev_values_from_metadata(self) -> Optional[torch.Tensor]:
        if not self.particles:
            return None
        prev_values = []
        for p in self.particles:
            if "prev_value" not in p.metadata:
                return None
            prev_values.append(p.metadata["prev_value"])
        return torch.tensor(prev_values, device=self._device, dtype=torch.float32)

    def _set_prev_values_metadata(self, values: torch.Tensor) -> None:
        values_list = values.detach().to("cpu").view(-1).tolist()
        for p, v in zip(self.particles, values_list):
            p.metadata["prev_value"] = float(v)
    
    def update_weights_with_twist(
        self,
        current_values: torch.Tensor,
        previous_values: torch.Tensor,
    ) -> None:
        """
        Update particle weights using twisted importance weights.
        
        Computes weight ratios based on value improvement and multiplies
        with current weights. Result is normalized.
        
        NOTE: This method only updates weights. Value state management
        (_previous_values, _current_values) is handled by step_with_twist()
        to avoid confusing double-assignment.
        
        Args:
            current_values: Value estimates at current step
            previous_values: Value estimates at previous step
        """
        if not self.config.use_twist:
            return
        
        current_values = current_values.to(self._device).float().view(-1)
        previous_values = previous_values.to(self._device).float().view(-1)

        if self.config.log_space:
            # Log-space: log_w_t = log_w_{t-1} + (logψ_t - logψ_{t-1})
            log_value_min = self.config.log_value_min
            current_values = torch.clamp(current_values, min=log_value_min, max=0.0)
            previous_values = torch.clamp(previous_values, min=log_value_min, max=0.0)

            delta = current_values - previous_values
            current_weights = self.get_weights()
            log_weights = torch.log(current_weights + self.config.epsilon) + delta
            log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
            new_weights = torch.exp(log_weights)
            self.set_weights(new_weights)
            return

        # Probability space: weight ratio = V_t / V_{t-1}
        weight_ratios = compute_twisted_weights(
            current_values,
            previous_values,
            epsilon=self.config.epsilon,
            log_space=False,
        )

        current_weights = self.get_weights()
        new_weights = current_weights * weight_ratios
        new_weights = normalize_weights(new_weights)
        self.set_weights(new_weights)
    
    def step_with_twist(self) -> None:
        """
        Single step of Twisted SMC.
        
        1. Compute values for current particles
        2. Update weights with twist (using previous values)
        3. Resample if ESS is low
        4. Update value state for next iteration
        
        Value State Management:
        This method is the sole owner of value state transitions.
        - _previous_values: values from previous step (used for weight ratios)
        - _current_values: values from current step (becomes previous next step)
        """
        if self.value_function is None:
            raise ValueError("Value function not set.")
        
        # Get current values
        current_values = self.compute_values().detach().to(self._device).view(-1)

        # Get previous values per particle (or initialize to current for first step)
        previous_values = self._get_prev_values_from_metadata()
        if previous_values is None:
            previous_values = current_values.clone().detach()

        # Update weights with twist
        self.update_weights_with_twist(current_values, previous_values)

        # Set prev values for next step (stored per particle for resampling safety)
        self._set_prev_values_metadata(current_values)
        self._previous_values = current_values.clone().detach()
        self._current_values = current_values.detach()

        # Adaptive resampling
        if self.should_resample():
            self.resample()
