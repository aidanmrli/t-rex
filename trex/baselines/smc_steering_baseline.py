"""
SMC Steering Baseline for Math Reasoning.

This module implements the Standard SMC Steering baseline for inference-time
compute scaling. It uses:
- vLLM for fast LLM generation
- PRM (Process Reward Model) for step-by-step scoring and resampling
- ORM (Outcome Reward Model) for final answer selection

Key Features:
- Particle filtering with adaptive resampling based on ESS
- SLURM checkpointing for preemptible clusters
- Time-based and problem-count based checkpointing
- WandB logging support
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
import torch

from trex.baselines.smc_config import SMCSteeringConfig, CheckpointManager
from trex.eval import MathVerifier

logger = logging.getLogger(__name__)


class SMCSteeringBaseline:
    """
    SMC Steering baseline with SLURM checkpointing support.
    
    This baseline implements inference-time compute scaling through:
    1. Initialize N particles with the problem prompt
    2. Generate reasoning steps with LLM (stop at "## Step N:")
    3. Score each step with PRM (Process Reward Model)
    4. Resample particles based on PRM scores when ESS is low
    5. Repeat until all particles finish or max steps reached
    6. Select best final answer using ORM (Outcome Reward Model)
    
    Attributes:
        config: SMCSteeringConfig with all parameters
        verifier: MathVerifier for answer verification
        generator: vLLM LLM instance (lazy loaded)
        reward_model: RewardModel for PRM/ORM (lazy loaded)
        checkpoint_mgr: CheckpointManager for SLURM support
    """
    
    def __init__(self, config: SMCSteeringConfig):
        """
        Initialize the SMC Steering Baseline.
        
        Args:
            config: SMCSteeringConfig with all parameters
        """
        self.config = config
        self.verifier = MathVerifier()
        self.generator = None
        self.reward_model = None
        
        self._shutdown_requested = False
        self._checkpoint_saved_on_exit = False
        self._last_checkpoint_time = time.time()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "generations"), exist_ok=True)
        
        # Initialize checkpoint manager
        if config.enable_checkpointing:
            checkpoint_path = os.path.join(config.output_dir, config.checkpoint_file)
            self.checkpoint_mgr = CheckpointManager(
                checkpoint_path, 
                config_hash=config.config_hash()
            )
            self._setup_signal_handlers()
        else:
            self.checkpoint_mgr = None
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown on SLURM preemption."""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\nReceived {sig_name}, saving checkpoint and exiting gracefully...")
            self._shutdown_requested = True
            if self.checkpoint_mgr and not self._checkpoint_saved_on_exit:
                self._checkpoint_saved_on_exit = True
                self.checkpoint_mgr.save()
            sys.exit(0)
        
        def atexit_handler():
            # Only save if signal handler didn't already save
            if self.checkpoint_mgr and not self._checkpoint_saved_on_exit:
                self._checkpoint_saved_on_exit = True
                self.checkpoint_mgr.save()
        
        # SIGTERM is sent by SLURM before preemption
        signal.signal(signal.SIGTERM, signal_handler)
        # SIGUSR1 is often used by SLURM for preemption warning
        signal.signal(signal.SIGUSR1, signal_handler)
        # Also save on normal exit
        atexit.register(atexit_handler)
        
        print("Signal handlers registered for graceful preemption handling.")
    
    def _init_generator(self) -> None:
        """Lazy initialization of vLLM generator."""
        if self.generator is None:
            from vllm import LLM
            
            print(f"Initializing vLLM generator with model: {self.config.generator_model_path}")
            self.generator = LLM(
                model=self.config.generator_model_path,
                tensor_parallel_size=self.config.generator_tp_size,
                max_num_seqs=self.config.max_num_seqs,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                seed=self.config.seed,
            )

    def _init_reward_model(self) -> None:
        """Lazy initialization of reward model."""
        if self.reward_model is None:
            from trex.models.reward_model import RewardModel
            
            print(f"Initializing reward model: {self.config.reward_model_path}")
            self.reward_model = RewardModel(
                model_path=self.config.reward_model_path,
                tp_size=self.config.reward_model_tp_size,
            )
    
    def _init_models(self) -> None:
        """Initialize all models."""
        self._init_generator()
        self._init_reward_model()
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        print(f"Loading dataset from {self.config.dataset_path}")
        dataset = []
        with jsonlines.open(self.config.dataset_path) as reader:
            for obj in reader:
                dataset.append(obj)
        print(f"Loaded {len(dataset)} problems")
        return dataset
    
    def prepare_prompt(self, problem: str) -> str:
        """
        Prepare a single problem with chat template.

        Args:
            problem: Raw problem text

        Returns:
            Formatted prompt ready for generation. In step mode this ends with
            "## Step 1:", while token mode returns the chat prompt without a step header.
        """
        if not self.config.apply_chat_template:
            if self.config.resampling_unit == "token":
                return problem
            return problem + "\n\n## Step 1:"

        self._init_generator()
        tokenizer = self.generator.get_tokenizer()

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": problem}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.config.resampling_unit == "token":
            return prompt

        # Add first step header so model continues from here
        # This prevents the model from outputting "## Step" at the start
        # which would trigger our stop string before any content
        return prompt + "\n\n## Step 1:"
    
    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SMC on a single problem.

        Args:
            item: Dataset item with prompt and answer keys

        Returns:
            Result dictionary with final answer, scores, and metrics
        """
        from trex.smc.llm_particle_filter import LLMParticleFilter

        self._init_models()

        problem = item[self.config.prompt_key]
        ground_truth = item[self.config.answer_key]
        prompt = self.prepare_prompt(problem)

        logger.debug(f"Evaluating problem: {problem[:100]}...")

        # Create particle filter
        pf = LLMParticleFilter(
            config=self.config,
            generator=self.generator,
            reward_model=self.reward_model,
        )

        # Initialize particles with the prompt
        pf.initialize(prompt)
        logger.debug(f"Initialized {self.config.n_particles} particles")

        # Run SMC loop
        start_time = time.time()
        best_particle = pf.run()
        elapsed_time = time.time() - start_time

        # Extract final answer from best particle
        final_text = best_particle.text
        extracted_answer = self.verifier.extract_answer(final_text)
        is_correct = self.verifier.verify(extracted_answer, ground_truth)

        # Get summary statistics
        summary = pf.get_summary()

        logger.debug(
            f"SMC completed: iterations={summary['smc_iteration']}, "
            f"reasoning_steps={best_particle.metadata.get('reasoning_step_count', 0)}, "
            f"orm_score={best_particle.metadata.get('orm_score', 0.0):.4f}, "
            f"correct={is_correct}, elapsed={elapsed_time:.2f}s"
        )

        result = {
            "problem": problem,
            "ground_truth": ground_truth,
            "final_text": final_text,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "orm_score": best_particle.metadata.get("orm_score", 0.0),
            "n_smc_iterations": summary["smc_iteration"],
            "n_reasoning_steps": best_particle.metadata.get("reasoning_step_count", 0),
            "elapsed_time": elapsed_time,
        }

        return result
    
    def _should_checkpoint(self, idx: int) -> bool:
        """Check if we should save a checkpoint."""
        current_time = time.time()
        
        # Problem-count based
        if (idx + 1) % self.config.checkpoint_interval == 0:
            return True
        
        # Time-based
        if current_time - self._last_checkpoint_time >= self.config.checkpoint_time_interval:
            return True
        
        return False
    
    def _save_checkpoint(self, idx: int, results: List[Dict[str, Any]]) -> None:
        """Save checkpoint with current progress."""
        if self.checkpoint_mgr:
            self.checkpoint_mgr.state["completed_idx"] = idx + 1
            self.checkpoint_mgr.state["results"] = results
            self.checkpoint_mgr.save()
            self._last_checkpoint_time = time.time()
            print(f"Checkpoint saved at problem {idx + 1}")
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution loop with checkpointing support.
        
        Returns:
            Summary dictionary with final metrics
        """
        # Check if already finished
        if self.checkpoint_mgr and self.checkpoint_mgr.is_finished():
            print("Experiment already completed (found finished checkpoint). Exiting.")
            return self._load_final_summary()
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Resume from checkpoint if exists
        start_idx = 0
        results = []
        if self.checkpoint_mgr:
            start_idx = self.checkpoint_mgr.get_resume_index()
            results = self.checkpoint_mgr.get_results()
            if start_idx > 0:
                print(f"Resuming from problem {start_idx}/{len(dataset)}")
        
        # Initialize WandB if requested
        if self.config.use_wandb:
            import wandb
            resume = "allow" if self.checkpoint_mgr else None
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
                resume=resume,
            )
        
        # Process each problem
        total_problems = len(dataset)
        for i in range(start_idx, total_problems):
            if self._shutdown_requested:
                print("Shutdown requested, saving progress...")
                break
            
            item = dataset[i]
            print(f"\nProcessing problem {i + 1}/{total_problems}")
            
            try:
                result = self.evaluate_single(item)
                results.append(result)
                
                # Log to WandB
                if self.config.use_wandb:
                    import wandb
                    wandb.log({
                        "problem_idx": i,
                        "is_correct": result["is_correct"],
                        "orm_score": result["orm_score"],
                        "n_smc_iterations": result["n_smc_iterations"],
                        "elapsed_time": result["elapsed_time"],
                        "running_accuracy": sum(r["is_correct"] for r in results) / len(results),
                    })
                
                # Print progress
                running_accuracy = sum(r["is_correct"] for r in results) / len(results)
                print(f"  Correct: {result['is_correct']}, "
                      f"ORM: {result['orm_score']:.3f}, "
                      f"Running Acc: {running_accuracy:.3f}")
                
            except Exception as e:
                logger.exception("Error processing problem %d", i + 1)
                print(f"  Error processing problem {i + 1}: {e}")
                results.append({
                    "problem": item[self.config.prompt_key],
                    "ground_truth": item[self.config.answer_key],
                    "error": str(e),
                    "is_correct": False,
                })
            
            # Checkpoint if needed
            if self._should_checkpoint(i):
                self._save_checkpoint(i, results)
        
        # Compute final metrics
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        avg_orm_score = np.mean([r.get("orm_score", 0.0) for r in results])
        avg_time = np.mean([r.get("elapsed_time", 0.0) for r in results])
        
        summary = {
            "config": asdict(self.config),
            "total_problems": total_count,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "avg_orm_score": float(avg_orm_score),
            "avg_time_per_problem": float(avg_time),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Save results
        self._save_results("summary", summary)
        self._save_generations("generations", results)
        
        # Log final metrics to WandB
        if self.config.use_wandb:
            import wandb
            wandb.log({
                "final/accuracy": accuracy,
                "final/avg_orm_score": avg_orm_score,
                "final/avg_time": avg_time,
            })
            wandb.finish()
        
        # Mark as finished
        if self.checkpoint_mgr:
            self.checkpoint_mgr.mark_finished()
        
        print(f"\n{'='*50}")
        print(f"Experiment completed!")
        print(f"  Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
        print(f"  Avg ORM Score: {avg_orm_score:.4f}")
        print(f"  Avg Time/Problem: {avg_time:.2f}s")
        print(f"{'='*50}")
        
        return summary
    
    def _save_results(self, name: str, data: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        path = os.path.join(self.config.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {name} to {path}")
    
    def _save_generations(self, name: str, results: List[Dict[str, Any]]) -> None:
        """Save generations to JSONL file."""
        path = os.path.join(self.config.output_dir, "generations", f"{name}.jsonl")
        with jsonlines.open(path, mode="w") as writer:
            for r in results:
                writer.write(r)
        print(f"Saved generations to {path}")
    
    def _load_final_summary(self) -> Dict[str, Any]:
        """Load final summary from saved results."""
        path = os.path.join(self.config.output_dir, "summary.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}


def main():
    """Main entry point for SMC Steering Baseline."""
    parser = argparse.ArgumentParser(
        description="SMC Steering Baseline for Math Reasoning"
    )
    
    # Model arguments
    parser.add_argument(
        "--generator_model_path", type=str, 
        default="Qwen/Qwen2.5-7B",
        help="Path to generator model"
    )
    parser.add_argument(
        "--reward_model_path", type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="Path to reward model (PRM/ORM)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_path", type=str,
        default="trex/data/gsm8k_platinum_test.jsonl",
        help="Path to dataset JSONL file"
    )
    
    # SMC arguments
    parser.add_argument(
        "--n_particles", type=int, default=16,
        help="Number of particles"
    )
    parser.add_argument(
        "--max_smc_iterations", type=int, default=20,
        help="Maximum SMC loop iterations (expand → score → resample)"
    )
    parser.add_argument(
        "--max_reasoning_steps", type=int, default=15,
        help="Maximum reasoning steps per particle (## Step N:)"
    )
    parser.add_argument(
        "--ess_threshold", type=float, default=0.5,
        help="ESS threshold for resampling"
    )
    parser.add_argument(
        "--resampling_method", type=str, default="systematic",
        choices=["multinomial", "systematic", "stratified"],
        help="Resampling method"
    )
    parser.add_argument(
        "--resampling_unit", type=str, default="step",
        choices=["step", "token"],
        help="Resampling unit: 'step' (## Step N:) or 'token' (fixed token chunks)"
    )
    parser.add_argument(
        "--resample_every_tokens", type=int, default=128,
        help="Token chunk size when resampling_unit='token'"
    )
    parser.add_argument(
        "--resampling_strategy", type=str, default="every_step",
        choices=["every_step", "ess_adaptive"],
        help="Resampling strategy: 'every_step' resamples after each SMC step, "
             "'ess_adaptive' only resamples when ESS drops below threshold"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_step_chunk_calls", type=int, default=4,
        help="Max generation chunks to complete a single reasoning step"
    )
    parser.add_argument(
        "--use_token_prompts", action="store_true",
        help="Use vLLM TokensPrompt with token IDs"
    )
    parser.add_argument(
        "--enable_prompt_truncation", action="store_true", default=True,
        help="Enable prompt truncation to fit model context"
    )
    parser.add_argument(
        "--disable_prompt_truncation", action="store_false", dest="enable_prompt_truncation",
        help="Disable prompt truncation"
    )
    parser.add_argument(
        "--prompt_max_tokens", type=int, default=None,
        help="Optional hard cap on prompt token length"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    
    # vLLM arguments
    parser.add_argument(
        "--generator_tp_size", type=int, default=1,
        help="Tensor parallelism for generator"
    )
    parser.add_argument(
        "--reward_model_tp_size", type=int, default=1,
        help="Tensor parallelism for reward model"
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory fraction for vLLM"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir", type=str,
        default="trex/results/smc_baseline",
        help="Output directory"
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable WandB logging"
    )
    
    # Checkpointing arguments
    parser.add_argument(
        "--enable_checkpointing", action="store_true", default=True,
        help="Enable checkpointing"
    )
    parser.add_argument(
        "--no_checkpointing", action="store_false", dest="enable_checkpointing",
        help="Disable checkpointing"
    )
    parser.add_argument(
        "--checkpoint_interval", type=int, default=5,
        help="Save checkpoint every N problems"
    )
    parser.add_argument(
        "--checkpoint_time_interval", type=int, default=600,
        help="Save checkpoint every N seconds"
    )

    # Logging arguments
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Use DEBUG for detailed SMC diagnostics."
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Optional file path to write logs to (in addition to stderr)"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    handlers = [logging.StreamHandler()]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    
    config = SMCSteeringConfig(
        generator_model_path=args.generator_model_path,
        reward_model_path=args.reward_model_path,
        dataset_path=args.dataset_path,
        n_particles=args.n_particles,
        max_smc_iterations=args.max_smc_iterations,
        max_reasoning_steps=args.max_reasoning_steps,
        ess_threshold=args.ess_threshold,
        resampling_method=args.resampling_method,
        resampling_unit=args.resampling_unit,
        resample_every_tokens=args.resample_every_tokens,
        resampling_strategy=args.resampling_strategy,
        temperature=args.temperature,
        max_step_chunk_calls=args.max_step_chunk_calls,
        use_token_prompts=args.use_token_prompts,
        enable_prompt_truncation=args.enable_prompt_truncation,
        prompt_max_tokens=args.prompt_max_tokens,
        seed=args.seed,
        generator_tp_size=args.generator_tp_size,
        reward_model_tp_size=args.reward_model_tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        enable_checkpointing=args.enable_checkpointing,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_time_interval=args.checkpoint_time_interval,
    )
    
    baseline = SMCSteeringBaseline(config)
    baseline.run()


if __name__ == "__main__":
    main()
