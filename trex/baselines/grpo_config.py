"""
Configuration for T-REX GRPO baseline experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GRPOConfig:
    """Configuration for GRPO baseline experiments using OpenRLHF."""
    
    # Model settings
    model_path: str = "Qwen/Qwen2.5-7B"
    
    # Dataset settings
    dataset: str = "gsm8k"  # "gsm8k" or "math"
    train_data_path: str = "trex/data/gsm8k_train.jsonl"
    eval_data_path: str = "trex/data/gsm8k_platinum_test.jsonl"
    input_key: str = "prompt"
    label_key: str = "label"
    
    # GRPO-specific parameters
    n_samples_per_prompt: int = 8      # Group size (G)
    init_kl_coef: float = 0.001        # KL penalty coefficient
    kl_estimator: str = "k3"           # k1, k2, k3 estimators
    
    # Training parameters
    train_batch_size: int = 128
    micro_train_batch_size: int = 4
    rollout_batch_size: int = 128
    micro_rollout_batch_size: int = 16
    max_epochs: int = 1
    actor_learning_rate: float = 5e-7
    
    # Generation parameters
    temperature: float = 1.0
    top_p: float = 1.0
    prompt_max_len: int = 1024
    generate_max_len: int = 2048
    
    # vLLM settings (for 4x H100)
    vllm_num_engines: int = 2
    vllm_tensor_parallel_size: int = 2
    vllm_gpu_memory_utilization: float = 0.5
    
    # Checkpointing (~30 min intervals)
    save_steps: int = 50
    load_checkpoint: bool = False
    ckpt_path: str = "./checkpoints"
    
    # Output settings
    output_dir: str = "trex/results/grpo_baseline"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "t-rex"
    wandb_run_name: Optional[str] = None
    
    # Chat template (for instruct models)
    apply_chat_template: bool = False
    
    # Reasoning mode (for models with thinking tokens)
    enable_thinking: bool = False
    
    def __post_init__(self):
        """Set derived values."""
        if self.wandb_run_name is None:
            self.wandb_run_name = f"grpo_{self.dataset}_n{self.n_samples_per_prompt}"
