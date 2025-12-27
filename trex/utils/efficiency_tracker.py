"""
Efficiency tracking for T-REX baseline comparisons.

This module provides utilities for measuring sample efficiency and
compute efficiency during training. It tracks:
- Samples generated (prompt-response pairs)
- Tokens generated
- Wall-clock time
- Accuracy at each evaluation point

Usage:
    tracker = EfficiencyTracker(thresholds=[0.5, 0.6, 0.7])
    
    # During training
    tracker.log_samples(num_samples, num_tokens)
    tracker.log_eval(accuracy)
    
    # After training
    tracker.save("results/efficiency_metrics.json")
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class EvalPoint:
    """A single evaluation checkpoint."""
    step: int
    accuracy: float
    cumulative_samples: int
    cumulative_tokens: int
    wall_time_seconds: float


@dataclass
class ThresholdResult:
    """Time-to-threshold results."""
    threshold: float
    samples_to_threshold: Optional[int] = None
    tokens_to_threshold: Optional[int] = None
    time_to_threshold_seconds: Optional[float] = None
    reached: bool = False


@dataclass
class EfficiencyMetrics:
    """Complete efficiency metrics for a training run."""
    method: str
    model: str
    dataset: str
    
    # Hyperparameters
    n_samples_per_prompt: int = 8
    batch_size: int = 128
    
    # Final results
    final_accuracy: float = 0.0
    total_samples: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    
    # Learning curve
    eval_points: List[EvalPoint] = field(default_factory=list)
    
    # Time-to-threshold
    threshold_results: List[ThresholdResult] = field(default_factory=list)
    
    # Area under learning curve (normalized)
    aulc: Optional[float] = None


class EfficiencyTracker:
    """
    Tracks sample efficiency and compute efficiency during training.
    
    This tracker is designed to work with any training method (GRPO, TSMC, T-REX)
    to enable fair comparisons of exploration efficiency.
    """
    
    def __init__(
        self,
        method: str,
        model: str,
        dataset: str,
        thresholds: List[float] = None,
        n_samples_per_prompt: int = 8,
        batch_size: int = 128,
    ):
        """
        Initialize the efficiency tracker.
        
        Args:
            method: Name of the method (e.g., "grpo", "tsmc", "trex")
            model: Model name (e.g., "Qwen2.5-7B")
            dataset: Dataset name (e.g., "gsm8k", "math500")
            thresholds: Accuracy thresholds for time-to-X% analysis
            n_samples_per_prompt: Group size for sample counting
            batch_size: Batch size for sample counting
        """
        self.thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        
        self.metrics = EfficiencyMetrics(
            method=method,
            model=model,
            dataset=dataset,
            n_samples_per_prompt=n_samples_per_prompt,
            batch_size=batch_size,
            threshold_results=[
                ThresholdResult(threshold=t) for t in self.thresholds
            ],
        )
        
        self._start_time = time.time()
        self._current_step = 0
        self._cumulative_samples = 0
        self._cumulative_tokens = 0
    
    def log_samples(self, num_prompts: int, tokens_generated: int):
        """
        Log samples generated during a training step.
        
        Args:
            num_prompts: Number of prompts in this batch
            tokens_generated: Total tokens generated in this batch
        """
        num_samples = num_prompts * self.metrics.n_samples_per_prompt
        self._cumulative_samples += num_samples
        self._cumulative_tokens += tokens_generated
        self._current_step += 1
    
    def log_eval(self, accuracy: float, step: Optional[int] = None):
        """
        Log an evaluation result.
        
        Args:
            accuracy: Accuracy on evaluation set (0.0 to 1.0)
            step: Optional step number (defaults to internal counter)
        """
        if step is not None:
            self._current_step = step
        
        elapsed = time.time() - self._start_time
        
        # Record eval point
        eval_point = EvalPoint(
            step=self._current_step,
            accuracy=accuracy,
            cumulative_samples=self._cumulative_samples,
            cumulative_tokens=self._cumulative_tokens,
            wall_time_seconds=elapsed,
        )
        self.metrics.eval_points.append(eval_point)
        
        # Check thresholds
        for result in self.metrics.threshold_results:
            if not result.reached and accuracy >= result.threshold:
                result.reached = True
                result.samples_to_threshold = self._cumulative_samples
                result.tokens_to_threshold = self._cumulative_tokens
                result.time_to_threshold_seconds = elapsed
                print(f"[Efficiency] Reached {result.threshold:.0%} accuracy at "
                      f"{self._cumulative_samples:,} samples, {elapsed:.1f}s")
        
        # Update final values
        self.metrics.final_accuracy = accuracy
        self.metrics.total_samples = self._cumulative_samples
        self.metrics.total_tokens = self._cumulative_tokens
        self.metrics.total_time_seconds = elapsed
    
    def compute_aulc(self) -> float:
        """
        Compute Area Under Learning Curve (AULC).
        
        AULC is normalized by the maximum possible area (accuracy=1.0 from start).
        Higher is better - indicates faster learning.
        
        Returns:
            AULC value between 0 and 1
        """
        if len(self.metrics.eval_points) < 2:
            return 0.0
        
        # Sort by samples
        points = sorted(self.metrics.eval_points, key=lambda p: p.cumulative_samples)
        
        # Compute area using trapezoidal rule
        area = 0.0
        max_samples = points[-1].cumulative_samples
        
        for i in range(1, len(points)):
            dx = points[i].cumulative_samples - points[i-1].cumulative_samples
            avg_y = (points[i].accuracy + points[i-1].accuracy) / 2
            area += dx * avg_y
        
        # Normalize by max possible area
        self.metrics.aulc = area / max_samples if max_samples > 0 else 0.0
        return self.metrics.aulc
    
    def get_summary(self) -> Dict:
        """Get a summary of efficiency metrics."""
        self.compute_aulc()
        
        summary = {
            "method": self.metrics.method,
            "model": self.metrics.model,
            "dataset": self.metrics.dataset,
            "final_accuracy": f"{self.metrics.final_accuracy:.2%}",
            "total_samples": f"{self.metrics.total_samples:,}",
            "total_tokens": f"{self.metrics.total_tokens:,}",
            "total_time": f"{self.metrics.total_time_seconds / 3600:.2f} hours",
            "aulc": f"{self.metrics.aulc:.4f}" if self.metrics.aulc else "N/A",
        }
        
        # Add threshold results
        for result in self.metrics.threshold_results:
            key = f"samples_to_{int(result.threshold * 100)}pct"
            if result.reached:
                summary[key] = f"{result.samples_to_threshold:,}"
            else:
                summary[key] = "not reached"
        
        return summary
    
    def save(self, path: str):
        """
        Save efficiency metrics to JSON file.
        
        Args:
            path: Output file path
        """
        self.compute_aulc()
        
        # Convert dataclasses to dicts
        data = asdict(self.metrics)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[Efficiency] Saved metrics to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EfficiencyMetrics":
        """Load efficiency metrics from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        # Reconstruct dataclasses
        data["eval_points"] = [EvalPoint(**p) for p in data["eval_points"]]
        data["threshold_results"] = [ThresholdResult(**r) for r in data["threshold_results"]]
        
        return EfficiencyMetrics(**data)


def compare_methods(metric_files: List[str]) -> Dict:
    """
    Compare efficiency metrics across multiple methods.
    
    Args:
        metric_files: List of paths to efficiency metric JSON files
    
    Returns:
        Comparison summary dict
    """
    results = []
    for path in metric_files:
        metrics = EfficiencyTracker.load(path)
        results.append(metrics)
    
    # Build comparison table
    comparison = {
        "methods": [m.method for m in results],
        "final_accuracy": [m.final_accuracy for m in results],
        "total_samples": [m.total_samples for m in results],
        "aulc": [m.aulc for m in results],
    }
    
    # Add threshold comparisons
    for threshold in [0.5, 0.6, 0.7]:
        key = f"samples_to_{int(threshold * 100)}pct"
        comparison[key] = []
        for m in results:
            result = next((r for r in m.threshold_results if r.threshold == threshold), None)
            if result and result.reached:
                comparison[key].append(result.samples_to_threshold)
            else:
                comparison[key].append(None)
    
    return comparison


if __name__ == "__main__":
    # Demo usage
    tracker = EfficiencyTracker(
        method="grpo",
        model="Qwen2.5-7B",
        dataset="gsm8k",
        thresholds=[0.3, 0.5, 0.7],
    )
    
    # Simulate training
    import random
    for step in range(100):
        tracker.log_samples(num_prompts=128, tokens_generated=128 * 8 * 512)
        if step % 10 == 0:
            acc = min(0.8, 0.1 + step * 0.007 + random.uniform(-0.02, 0.02))
            tracker.log_eval(acc, step=step)
    
    # Print summary
    print("\nEfficiency Summary:")
    for k, v in tracker.get_summary().items():
        print(f"  {k}: {v}")
    
    # Save
    tracker.save("/tmp/grpo_efficiency.json")
