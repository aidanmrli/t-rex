# OpenRLHF-compatible reward function for T-REX GRPO baseline.
# Loaded via: --remote_rm_url trex/baselines/grpo_reward_func.py

"""
Math reward function for GRPO training with OpenRLHF.

This module provides a reward function compatible with OpenRLHF's
`--remote_rm_url` interface. It reuses the MathVerifier from T-REX
to verify mathematical answers and compute rewards.

Features:
- Real-time sample/token counting
- WandB integration for live efficiency tracking
- Threshold detection (logs when X% accuracy is reached)
- JSON persistence for post-hoc analysis
"""

import atexit
import json
import os
import time
import torch
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Import T-REX components
from trex.eval import MathVerifier
from trex.eval.parser import extract_last_boxed

# Try to import wandb (optional dependency)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# =============================================================================
# Efficiency Tracker (Singleton)
# =============================================================================

@dataclass
class ThresholdResult:
    """Time-to-threshold results."""
    threshold: float
    samples_to_threshold: Optional[int] = None
    tokens_to_threshold: Optional[int] = None
    time_to_threshold_seconds: Optional[float] = None
    step_reached: Optional[int] = None
    reached: bool = False


@dataclass
class EfficiencyState:
    """Persistent state for efficiency tracking."""
    method: str = "grpo"
    model: str = ""
    dataset: str = ""
    
    # Counters
    total_samples: int = 0
    total_tokens: int = 0
    total_reward_calls: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    
    # Best accuracy seen
    best_accuracy: float = 0.0
    
    # Threshold tracking
    thresholds: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    threshold_results: Dict[float, ThresholdResult] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize threshold results
        for t in self.thresholds:
            if t not in self.threshold_results:
                self.threshold_results[t] = ThresholdResult(threshold=t)


class EfficiencyTracker:
    """
    Singleton tracker for sample efficiency during GRPO training.
    
    This tracker is automatically initialized on first reward_func call
    and logs to WandB in real-time.
    """
    
    _instance: Optional["EfficiencyTracker"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.state = EfficiencyState()
        self._initialized = True
        self._save_path: Optional[str] = None
        
        # Register save on exit
        atexit.register(self._save_on_exit)
        
        print("[EfficiencyTracker] Initialized")
    
    def configure(
        self,
        method: str = "grpo",
        model: str = "",
        dataset: str = "",
        save_path: Optional[str] = None,
        thresholds: Optional[List[float]] = None,
    ):
        """Configure the tracker (call once at start of training)."""
        self.state.method = method
        self.state.model = model
        self.state.dataset = dataset
        self._save_path = save_path
        
        if thresholds:
            self.state.thresholds = thresholds
            self.state.threshold_results = {
                t: ThresholdResult(threshold=t) for t in thresholds
            }
        
        print(f"[EfficiencyTracker] Configured: method={method}, model={model}, dataset={dataset}")
    
    def log_batch(
        self,
        num_samples: int,
        num_tokens: int,
        accuracy: float,
        extra_metrics: Optional[Dict] = None,
    ):
        """
        Log a batch of samples and update efficiency metrics.
        
        Args:
            num_samples: Number of (prompt, response) pairs in this batch
            num_tokens: Estimated tokens generated in this batch
            accuracy: Accuracy on this batch
            extra_metrics: Additional metrics to log to WandB
        """
        self.state.total_samples += num_samples
        self.state.total_tokens += num_tokens
        self.state.total_reward_calls += 1
        
        elapsed = time.time() - self.state.start_time
        
        # Update best accuracy (use EMA for stability)
        alpha = 0.1  # Smoothing factor
        if self.state.best_accuracy == 0:
            self.state.best_accuracy = accuracy
        else:
            self.state.best_accuracy = max(
                self.state.best_accuracy,
                alpha * accuracy + (1 - alpha) * self.state.best_accuracy
            )
        
        # Check thresholds
        for threshold, result in self.state.threshold_results.items():
            if not result.reached and accuracy >= threshold:
                result.reached = True
                result.samples_to_threshold = self.state.total_samples
                result.tokens_to_threshold = self.state.total_tokens
                result.time_to_threshold_seconds = elapsed
                result.step_reached = self.state.total_reward_calls
                
                print(f"[EfficiencyTracker] 🎯 Reached {threshold:.0%} accuracy!")
                print(f"    Samples: {self.state.total_samples:,}")
                print(f"    Tokens: {self.state.total_tokens:,}")
                print(f"    Time: {elapsed:.1f}s")
                
                # Log milestone to WandB
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        f"efficiency/samples_to_{int(threshold*100)}pct": self.state.total_samples,
                        f"efficiency/tokens_to_{int(threshold*100)}pct": self.state.total_tokens,
                        f"efficiency/time_to_{int(threshold*100)}pct": elapsed,
                    })
        
        # Build metrics dict
        metrics = {
            "efficiency/cumulative_samples": self.state.total_samples,
            "efficiency/cumulative_tokens": self.state.total_tokens,
            "efficiency/batch_accuracy": accuracy,
            "efficiency/best_accuracy": self.state.best_accuracy,
            "efficiency/elapsed_seconds": elapsed,
            "efficiency/samples_per_second": self.state.total_samples / elapsed if elapsed > 0 else 0,
        }
        
        if extra_metrics:
            for k, v in extra_metrics.items():
                metrics[f"efficiency/{k}"] = v
        
        # Log to WandB
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics)
        
        # Periodic save (every 100 calls)
        if self._save_path and self.state.total_reward_calls % 100 == 0:
            self.save()
    
    def get_summary(self) -> Dict:
        """Get current efficiency summary."""
        elapsed = time.time() - self.state.start_time
        
        summary = {
            "method": self.state.method,
            "model": self.state.model,
            "dataset": self.state.dataset,
            "total_samples": self.state.total_samples,
            "total_tokens": self.state.total_tokens,
            "best_accuracy": f"{self.state.best_accuracy:.2%}",
            "elapsed_hours": f"{elapsed / 3600:.2f}",
            "samples_per_second": f"{self.state.total_samples / elapsed:.1f}" if elapsed > 0 else "N/A",
        }
        
        # Add threshold results
        for threshold, result in self.state.threshold_results.items():
            key = f"samples_to_{int(threshold * 100)}pct"
            if result.reached:
                summary[key] = result.samples_to_threshold
            else:
                summary[key] = None
        
        return summary
    
    def save(self, path: Optional[str] = None):
        """Save efficiency metrics to JSON."""
        save_path = path or self._save_path
        if not save_path:
            return
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "method": self.state.method,
            "model": self.state.model,
            "dataset": self.state.dataset,
            "total_samples": self.state.total_samples,
            "total_tokens": self.state.total_tokens,
            "total_reward_calls": self.state.total_reward_calls,
            "best_accuracy": self.state.best_accuracy,
            "elapsed_seconds": time.time() - self.state.start_time,
            "thresholds": {
                f"{int(t*100)}pct": asdict(r) 
                for t, r in self.state.threshold_results.items()
            },
        }
        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[EfficiencyTracker] Saved to {save_path}")
    
    def _save_on_exit(self):
        """Save metrics when process exits."""
        if self._save_path:
            self.save()
            print(f"[EfficiencyTracker] Final summary: {self.get_summary()}")


# Global tracker instance
_tracker: Optional[EfficiencyTracker] = None


def get_tracker() -> EfficiencyTracker:
    """Get the global efficiency tracker."""
    global _tracker
    if _tracker is None:
        _tracker = EfficiencyTracker()
        
        # Try to configure from environment variables
        _tracker.configure(
            method=os.environ.get("TREX_METHOD", "grpo"),
            model=os.environ.get("TREX_MODEL", ""),
            dataset=os.environ.get("TREX_DATASET", ""),
            save_path=os.environ.get("TREX_EFFICIENCY_PATH", None),
        )
    
    return _tracker


# =============================================================================
# Reward Function
# =============================================================================

# Global verifier instance (lazy initialization)
_verifier: Optional[MathVerifier] = None


def get_verifier() -> MathVerifier:
    """Get or create the global MathVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = MathVerifier()
    return _verifier


def extract_response(query: str, prompt: str) -> str:
    """
    Extract the model's response from the full query.
    
    Args:
        query: Full sequence (prompt + response)
        prompt: The input prompt
    
    Returns:
        The response portion only
    """
    # Simple extraction: response is everything after the prompt
    if query.startswith(prompt):
        return query[len(prompt):]
    
    # Fallback: try to find common assistant markers
    markers = [
        "<|im_start|>assistant\n",
        "<|assistant|>",
        "Assistant:",
    ]
    for marker in markers:
        if marker in query:
            return query.split(marker)[-1]
    
    # Last resort: return the full query
    return query


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4


def reward_func(queries: List[str], prompts: List[str], labels: List[str], **kwargs) -> dict:
    """
    OpenRLHF-compatible reward function for math reasoning.
    
    This function is called by OpenRLHF during GRPO training to compute
    rewards for generated responses. It also tracks efficiency metrics
    and logs to WandB.
    
    Args:
        queries: List of full sequences (prompt + response)
        prompts: List of input prompts
        labels: List of ground truth answers
        **kwargs: Additional arguments (ignored)
    
    Returns:
        dict with:
            - rewards: Tensor of reward values for advantage computation
            - scores: Tensor of scores for dynamic filtering (0-1 range)
            - extra_logs: Dict of additional metrics for WandB logging
    """
    verifier = get_verifier()
    tracker = get_tracker()
    
    rewards = []
    correctness = []
    has_boxed_count = 0
    total_tokens = 0
    
    for query, prompt, label in zip(queries, prompts, labels):
        # Extract the response from the full query
        response = extract_response(query, prompt)
        
        # Estimate tokens for efficiency tracking
        total_tokens += estimate_tokens(response)
        
        # Verify correctness using MathVerifier
        correct = verifier.verify(response, label)
        
        # Check if response has proper \boxed{} formatting
        has_boxed = extract_last_boxed(response) is not None
        if has_boxed:
            has_boxed_count += 1
        
        # Compute reward with format penalty
        # Reward shaping:
        #   +1.0 for correct answer
        #   -1.0 for no \boxed{} (format penalty)
        #    0.0 for incorrect but properly formatted
        if correct:
            score = 1.0
        elif not has_boxed:
            score = -1.0  # Format penalty
        else:
            score = 0.0
        
        rewards.append(score)
        correctness.append(float(correct))
    
    # Compute aggregate metrics
    n = len(queries)
    accuracy = sum(correctness) / n if n > 0 else 0.0
    format_rate = has_boxed_count / n if n > 0 else 0.0
    mean_reward = sum(rewards) / n if n > 0 else 0.0
    
    # Log to efficiency tracker (with WandB integration)
    tracker.log_batch(
        num_samples=n,
        num_tokens=total_tokens,
        accuracy=accuracy,
        extra_metrics={
            "format_rate": format_rate,
            "mean_reward": mean_reward,
        }
    )
    
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "scores": torch.tensor(rewards, dtype=torch.float32),
        "extra_logs": {
            "accuracy": accuracy,
            "format_rate": format_rate,
            "mean_reward": mean_reward,
            "cumulative_samples": tracker.state.total_samples,
            "cumulative_tokens": tracker.state.total_tokens,
        }
    }


if __name__ == "__main__":
    # Test the reward function with efficiency tracking
    import os
    os.environ["TREX_METHOD"] = "grpo"
    os.environ["TREX_MODEL"] = "Qwen2.5-7B"
    os.environ["TREX_DATASET"] = "gsm8k"
    os.environ["TREX_EFFICIENCY_PATH"] = "/tmp/test_efficiency.json"
    
    # Simulate multiple batches
    for batch in range(5):
        test_queries = [
            "What is 2+2? The answer is \\boxed{4}",
            "What is 3+3? The answer is 6",  # No boxed
            "What is 5+5? The answer is \\boxed{10}",  # Correct!
        ]
        test_prompts = ["What is 2+2?", "What is 3+3?", "What is 5+5?"]
        test_labels = ["4", "6", "10"]
        
        result = reward_func(test_queries, test_prompts, test_labels)
        print(f"Batch {batch + 1}:")
        print(f"  Rewards: {result['rewards'].tolist()}")
        print(f"  Cumulative samples: {result['extra_logs']['cumulative_samples']}")
    
    # Print final summary
    tracker = get_tracker()
    print(f"\nFinal Summary: {tracker.get_summary()}")

