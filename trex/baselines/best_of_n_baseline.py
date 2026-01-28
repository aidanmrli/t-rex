import argparse
import atexit
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
import torch
from vllm import LLM, SamplingParams

from trex.baselines.config import BaselineConfig
from trex.eval import MathVerifier


class CheckpointManager:
    """Manages checkpointing for preemptible cluster environments."""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.state: Dict[str, Any] = {
            "phase": "sweep",  # "sweep" or "full_eval"
            "sweep_completed_temps": {},  # {temp: metrics}
            "sweep_best_temp": None,
            "sweep_best_val": -1.0,
            "full_eval_completed_idx": 0,  # Number of problems completed
            "full_eval_results": [],  # List of per-problem results
            "finished": False,
        }
        self._load()
    
    def _load(self):
        """Load existing checkpoint if available."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as f:
                    saved_state = json.load(f)
                self.state.update(saved_state)
                print(f"Loaded checkpoint from {self.checkpoint_path}")
                print(f"  Phase: {self.state['phase']}")
                if self.state['phase'] == 'sweep':
                    print(f"  Completed temperatures: {list(self.state['sweep_completed_temps'].keys())}")
                else:
                    print(f"  Completed problems: {self.state['full_eval_completed_idx']}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load checkpoint ({e}), starting fresh.")
    
    def save(self):
        """Save current state to checkpoint file."""
        # Write to temp file first, then rename for atomicity
        temp_path = self.checkpoint_path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(self.state, f, indent=2)
        os.rename(temp_path, self.checkpoint_path)
        print(f"Checkpoint saved: phase={self.state['phase']}, "
              f"sweep_temps={len(self.state['sweep_completed_temps'])}, "
              f"eval_idx={self.state['full_eval_completed_idx']}")
    
    def is_finished(self) -> bool:
        return self.state.get("finished", False)
    
    def mark_finished(self):
        self.state["finished"] = True
        self.save()


class BestOfNBaseline:
    """
    Brute-force rejection sampling baseline for math reasoning.
    
    Workflow:
    1. Load dataset (GSM8K or MATH).
    2. Temperature sweep on first K problems to find optimal temperature for pass@N.
    3. Full evaluation on the entire dataset using the best temperature.
    4. Save metrics and generations.
    
    Supports checkpointing for preemptible clusters.
    """

    def __init__(self, config: BaselineConfig):
        self.config = config
        self.verifier = MathVerifier()
        self.llm: Optional[LLM] = None
        self._shutdown_requested = False
        self._checkpoint_saved_on_exit = False  # Prevent double-save
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "generations"), exist_ok=True)
        
        # Initialize checkpoint manager
        if self.config.enable_checkpointing:
            checkpoint_path = os.path.join(self.config.output_dir, self.config.checkpoint_file)
            self.checkpoint_mgr = CheckpointManager(checkpoint_path)
            self._setup_signal_handlers()
        else:
            self.checkpoint_mgr = None
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown on preemption."""
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
        # Also save on normal exit (but only if signal handler didn't already save)
        atexit.register(atexit_handler)
        
        print("Signal handlers registered for graceful preemption handling.")

    def _init_llm(self):
        """Lazy initialization of vLLM engine."""
        if self.llm is None:
            print(f"Initializing vLLM with model: {self.config.model_path}")
            
            # Prepare extra arguments for reasoning
            extra_kwargs = {}
            if self.config.enable_reasoning:
                extra_kwargs["enable_reasoning"] = True
                if self.config.reasoning_parser:
                    extra_kwargs["reasoning_parser"] = self.config.reasoning_parser

            self.llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tp_size,
                max_num_seqs=self.config.max_num_seqs,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                seed=self.config.seed,
                **extra_kwargs,
            )

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file."""
        print(f"Loading dataset from {self.config.dataset_path}")
        dataset = []
        with jsonlines.open(self.config.dataset_path) as reader:
            for obj in reader:
                dataset.append(obj)
        return dataset

    def prepare_prompts(self, dataset: List[Dict[str, Any]]) -> List[str]:
        """Apply chat template if configured."""
        prompts = []
        
        # Initialize LLM early if we need the tokenizer for chat templates
        if self.config.apply_chat_template:
            self._init_llm()
            tokenizer = self.llm.get_tokenizer()
            
        for item in dataset:
            prompt = item[self.config.prompt_key]
            if self.config.apply_chat_template:
                # Construct chat messages
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                # Use vLLM's tokenizer to apply the chat template
                kwargs = {}
                if self.config.enable_thinking:
                    kwargs["enable_thinking"] = True
                    
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    **kwargs
                )
                prompts.append(formatted_prompt)
            else:
                prompts.append(prompt)
        return prompts

    def generate_samples(self, prompts: List[str], temperature: float, n: int) -> List[Any]:
        """Generate N samples for each prompt using vLLM."""
        self._init_llm()

        # vLLM requires n=1 for greedy sampling (temperature=0)
        effective_n = 1 if temperature == 0.0 else n

        sampling_params = SamplingParams(
            n=effective_n,
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )
        
        print(f"Generating {effective_n} samples per prompt at temperature {temperature}...")
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        return outputs

    def compute_metrics(self, outputs: List[Any], ground_truths: List[str], n_values: List[int]) -> Dict[str, float]:
        """
        Compute pass@k for various k values.
        
        pass@k = 1 - E[ (n-c choose k) / (n choose k) ]
        where n is total samples, c is correct samples.
        """
        results = {f"pass@{k}": [] for k in n_values}
        
        for output, gold in zip(outputs, ground_truths):
            # Extract texts from vLLM RequestOutput
            samples = [o.text for o in output.outputs]
            
            # Verify each sample
            correct_mask = [self.verifier.verify(s, gold) for s in samples]
            c = sum(correct_mask)
            n = len(samples)
            
            for k in n_values:
                if k > n:
                    continue
                
                # Probability of failure: (n-c choose k) / (n choose k)
                if n - c < k:
                    pass_at_k = 1.0
                else:
                    # Use log-gamma for numerical stability
                    from scipy.special import comb
                    # prob_fail = comb(n - c, k) / comb(n, k)
                    # For speed/accuracy:
                    import math
                    def lcomb(n, k):
                        if k < 0 or k > n: return -float('inf')
                        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    
                    log_prob_fail = lcomb(n - c, k) - lcomb(n, k)
                    pass_at_k = 1.0 - math.exp(log_prob_fail)
                
                results[f"pass@{k}"].append(pass_at_k)
        
        # Handle empty lists (when k > n for all samples) - return NaN for unavailable metrics
        return {k: float(np.mean(v)) if v else float('nan') for k, v in results.items()}
    
    def compute_per_problem_results(self, output: Any, gold: str, n_values: List[int]) -> Dict[str, Any]:
        """Compute results for a single problem (for checkpointing)."""
        samples = [o.text for o in output.outputs]
        correct_mask = [self.verifier.verify(s, gold) for s in samples]
        c = sum(correct_mask)
        n = len(samples)
        
        pass_at_k_results = {}
        for k in n_values:
            if k > n:
                continue
            if n - c < k:
                pass_at_k = 1.0
            else:
                import math
                def lcomb(n, k):
                    if k < 0 or k > n: return -float('inf')
                    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                log_prob_fail = lcomb(n - c, k) - lcomb(n, k)
                pass_at_k = 1.0 - math.exp(log_prob_fail)
            pass_at_k_results[f"pass@{k}"] = pass_at_k
        
        return {
            "samples": samples,
            "correct_mask": correct_mask,
            "num_correct": c,
            "pass_at_k": pass_at_k_results,
        }

    def save_results(self, name: str, data: Dict[str, Any]):
        """Save results to JSON file."""
        path = os.path.join(self.config.output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {name} to {path}")

    def save_generations(self, name: str, outputs: List[Any], dataset: List[Dict[str, Any]], temperature: float):
        """Save raw generations to JSONL."""
        path = os.path.join(self.config.output_dir, "generations", f"{name}.jsonl")
        with jsonlines.open(path, mode="w") as writer:
            for output, item in zip(outputs, dataset):
                record = {
                    "prompt": item[self.config.prompt_key],
                    "ground_truth": item[self.config.answer_key],
                    "temperature": temperature,
                    "samples": [o.text for o in output.outputs]
                }
                writer.write(record)
        print(f"Saved generations to {path}")
    
    def save_generations_from_checkpoint(self, name: str, results: List[Dict], dataset: List[Dict[str, Any]], temperature: float):
        """Save generations from checkpointed results."""
        path = os.path.join(self.config.output_dir, "generations", f"{name}.jsonl")
        with jsonlines.open(path, mode="w") as writer:
            for result, item in zip(results, dataset):
                record = {
                    "prompt": item[self.config.prompt_key],
                    "ground_truth": item[self.config.answer_key],
                    "temperature": temperature,
                    "samples": result["samples"]
                }
                writer.write(record)
        print(f"Saved generations to {path}")

    def run_temperature_sweep(self, sweep_prompts: List[str], sweep_gts: List[str]) -> float:
        """Run temperature sweep with checkpointing support."""
        state = self.checkpoint_mgr.state if self.checkpoint_mgr else {}
        completed_temps = state.get("sweep_completed_temps", {})
        best_temp = state.get("sweep_best_temp") or self.config.temperatures[0]
        best_val = state.get("sweep_best_val", -1.0)
        
        metrics_to_track = [1, self.config.n_samples]
        sweep_results = dict(completed_temps)  # Start with already completed
        
        for temp in self.config.temperatures:
            # Skip already completed temperatures
            if str(temp) in completed_temps:
                print(f"Skipping temperature {temp} (already completed)")
                metrics = completed_temps[str(temp)]
                current_val = metrics[f"pass@{self.config.n_samples}"]
                if current_val > best_val:
                    best_val = current_val
                    best_temp = temp
                continue
            
            # Check for shutdown request
            if self._shutdown_requested:
                print("Shutdown requested, saving progress...")
                break
            
            outputs = self.generate_samples(sweep_prompts, temp, self.config.n_samples)
            metrics = self.compute_metrics(outputs, sweep_gts, metrics_to_track)
            sweep_results[str(temp)] = metrics
            
            print(f"Temp {temp}: {metrics}")
            
            current_val = metrics[f"pass@{self.config.n_samples}"]
            if current_val > best_val:
                best_val = current_val
                best_temp = temp
            
            # Update checkpoint after each temperature
            if self.checkpoint_mgr:
                self.checkpoint_mgr.state["sweep_completed_temps"] = sweep_results
                self.checkpoint_mgr.state["sweep_best_temp"] = best_temp
                self.checkpoint_mgr.state["sweep_best_val"] = best_val
                self.checkpoint_mgr.save()
            
            if self.config.use_wandb:
                import wandb
                log_data = {f"sweep/temp_{temp}/{k}": v for k, v in metrics.items()}
                wandb.log(log_data)
        
        print(f"\nBest temperature selected: {best_temp} (pass@{self.config.n_samples} = {best_val:.4f})")
        self.save_results("sweep_results", {
            "best_temp": best_temp,
            "sweep_metrics": sweep_results
        })
        
        return best_temp
    
    def run_full_evaluation(self, prompts: List[str], ground_truths: List[str], 
                            dataset: List[Dict[str, Any]], best_temp: float) -> Dict[str, float]:
        """Run full evaluation with chunked checkpointing."""
        state = self.checkpoint_mgr.state if self.checkpoint_mgr else {}
        start_idx = state.get("full_eval_completed_idx", 0)
        results = state.get("full_eval_results", [])
        
        # Compute k values for metrics
        k_values = [1, 2, 4, 8]
        if self.config.n_samples not in k_values:
            k_values.append(self.config.n_samples)
        k_values = sorted(list(set(k_values)))
        
        total_problems = len(prompts)
        chunk_size = self.config.eval_chunk_size
        
        print(f"Starting full evaluation from problem {start_idx}/{total_problems}")
        
        # Process in chunks
        for chunk_start in range(start_idx, total_problems, chunk_size):
            if self._shutdown_requested:
                print("Shutdown requested, saving progress...")
                break
            
            chunk_end = min(chunk_start + chunk_size, total_problems)
            chunk_prompts = prompts[chunk_start:chunk_end]
            chunk_gts = ground_truths[chunk_start:chunk_end]
            
            print(f"\nProcessing problems {chunk_start+1}-{chunk_end}/{total_problems}")
            
            # Generate samples for this chunk
            outputs = self.generate_samples(chunk_prompts, best_temp, self.config.n_samples)
            
            # Process each problem in the chunk
            for i, (output, gold) in enumerate(zip(outputs, chunk_gts)):
                problem_result = self.compute_per_problem_results(output, gold, k_values)
                results.append(problem_result)
            
            # Update checkpoint after each chunk
            if self.checkpoint_mgr:
                self.checkpoint_mgr.state["phase"] = "full_eval"
                self.checkpoint_mgr.state["full_eval_completed_idx"] = chunk_end
                self.checkpoint_mgr.state["full_eval_results"] = results
                self.checkpoint_mgr.save()
        
        # Aggregate final metrics
        final_metrics = {f"pass@{k}": [] for k in k_values}
        for result in results:
            for k in k_values:
                key = f"pass@{k}"
                if key in result["pass_at_k"]:
                    final_metrics[key].append(result["pass_at_k"][key])
        
        final_metrics = {k: float(np.mean(v)) for k, v in final_metrics.items() if v}
        
        return final_metrics, results

    def run(self):
        """Main execution flow with checkpointing support."""
        # Check if already finished
        if self.checkpoint_mgr and self.checkpoint_mgr.is_finished():
            print("Experiment already completed (found finished checkpoint). Exiting.")
            return
        
        dataset = self.load_dataset()
        prompts = self.prepare_prompts(dataset)
        ground_truths = [item[self.config.answer_key] for item in dataset]
        
        # Initialize WandB if requested
        if self.config.use_wandb:
            import wandb
            # Try to resume existing run if checkpointing
            resume = "allow" if self.checkpoint_mgr else None
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
                resume=resume
            )

        # Determine current phase from checkpoint
        current_phase = "sweep"
        if self.checkpoint_mgr:
            current_phase = self.checkpoint_mgr.state.get("phase", "sweep")
        
        # Phase 1: Temperature Sweep
        if current_phase == "sweep":
            print(f"\n--- Phase 1: Temperature Sweep (first {self.config.sweep_size} problems) ---")
            sweep_prompts = prompts[:self.config.sweep_size]
            sweep_gts = ground_truths[:self.config.sweep_size]
            
            best_temp = self.run_temperature_sweep(sweep_prompts, sweep_gts)
            
            # Transition to full eval phase
            if self.checkpoint_mgr:
                self.checkpoint_mgr.state["phase"] = "full_eval"
                self.checkpoint_mgr.state["sweep_best_temp"] = best_temp
                self.checkpoint_mgr.save()
        else:
            # Resume: get best temp from checkpoint
            best_temp = self.checkpoint_mgr.state.get("sweep_best_temp")
            print(f"\nResuming from checkpoint. Best temperature from sweep: {best_temp}")

        # Check for shutdown before starting full eval
        if self._shutdown_requested:
            return

        # Phase 2: Full Evaluation
        print(f"\n--- Phase 2: Full Evaluation (Temp={best_temp}) ---")
        final_metrics, results = self.run_full_evaluation(prompts, ground_truths, dataset, best_temp)
        
        print(f"Final Metrics: {final_metrics}")
        
        summary = {
            "config": asdict(self.config),
            "best_temp": best_temp,
            "final_metrics": final_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_results("summary", summary)
        
        if self.config.save_generations:
            self.save_generations_from_checkpoint(f"full_eval_temp_{best_temp}", results, dataset, best_temp)
            
        if self.config.use_wandb:
            import wandb
            wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
            wandb.finish()
        
        # Mark as finished
        if self.checkpoint_mgr:
            self.checkpoint_mgr.mark_finished()
        
        print("\nExperiment completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Best-of-N Baseline for Math Reasoning")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--dataset_path", type=str, default="trex/data/gsm8k_platinum_test.jsonl")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--sweep_size", type=int, default=100)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Max sequences in KV cache")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory fraction for vLLM")
    parser.add_argument("--output_dir", type=str, default="trex/results/bon_baseline")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true", default=True, 
                        help="Whether to apply chat template (default: True)")
    parser.add_argument("--no_chat_template", action="store_false", dest="apply_chat_template",
                        help="Disable chat template for base models")
    # Checkpointing arguments
    parser.add_argument("--enable_checkpointing", action="store_true", default=True,
                        help="Enable checkpointing for preemptible clusters (default: True)")
    parser.add_argument("--no_checkpointing", action="store_false", dest="enable_checkpointing",
                        help="Disable checkpointing")
    parser.add_argument("--eval_chunk_size", type=int, default=50,
                        help="Number of problems per checkpoint during full evaluation")

    # Reasoning arguments
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking in chat template")
    parser.add_argument("--enable_reasoning", action="store_true", help="Enable reasoning in vLLM")
    parser.add_argument("--reasoning_parser", type=str, default=None, help="Parser for reasoning tokens (e.g. deepseek_r1)")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None, help="List of temperatures to sweep")
    
    args = parser.parse_args()
    
    config = BaselineConfig(
        model_path=args.model_path,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        n_samples=args.n_samples,
        sweep_size=args.sweep_size,
        tp_size=args.tp_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        apply_chat_template=args.apply_chat_template,
        enable_checkpointing=args.enable_checkpointing,
        eval_chunk_size=args.eval_chunk_size,
        enable_thinking=args.enable_thinking,
        enable_reasoning=args.enable_reasoning,
        reasoning_parser=args.reasoning_parser,
    )
    
    if args.temperatures:
        config.temperatures = args.temperatures
    
    baseline = BestOfNBaseline(config)
    baseline.run()


if __name__ == "__main__":
    main()
