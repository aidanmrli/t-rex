"""
Configuration for T-REX baseline experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaselineConfig:
    """Configuration for Best-of-N baseline experiments."""
    
    # Model settings
    model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    
    # Dataset settings
    dataset: str = "gsm8k"  # "gsm8k" or "math"
    dataset_path: str = "trex/data/gsm8k_test.jsonl"
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    
    # Sampling parameters
    temperatures: List[float] = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2])
    n_samples: int = 8  # Best-of-N value
    max_new_tokens: int = 2048
    top_p: float = 1.0
    
    # Experiment structure
    sweep_size: int = 100  # Number of problems for temperature sweep
    seed: int = 42
    
    # vLLM settings
    tp_size: int = 1  # Tensor parallelism
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    
    # Output settings
    output_dir: str = "trex/results/bon_baseline"
    save_generations: bool = True
    
    # Checkpointing (for preemptible clusters)
    enable_checkpointing: bool = True
    checkpoint_file: str = "checkpoint.json"
    eval_chunk_size: int = 100  # Save checkpoint every N problems during full eval
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "t-rex"
    wandb_run_name: Optional[str] = None
    
    # Chat template (for instruct models)
    apply_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant that solves math problems step by step. Put your final answer in \\boxed{}."
    
    # Reasoning / Thinking mode
    enable_thinking: bool = False  # Enable thinking in chat template
    enable_reasoning: bool = False  # Enable reasoning in vLLM engine
    reasoning_parser: Optional[str] = None  # Parser for reasoning tokens (e.g. "deepseek_r1")
    
    def __post_init__(self):
        """Set derived values."""
        if self.wandb_run_name is None:
            self.wandb_run_name = f"bon_{self.dataset}_n{self.n_samples}"
