import argparse
import json
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import jsonlines
import numpy as np
import torch
from vllm import LLM, SamplingParams

from trex.baselines.config import BaselineConfig
from trex.eval import MathVerifier


class BestOfNBaseline:
    """
    Brute-force rejection sampling baseline for math reasoning.
    
    Workflow:
    1. Load dataset (GSM8K or MATH).
    2. Temperature sweep on first K problems to find optimal temperature for pass@N.
    3. Full evaluation on the entire dataset using the best temperature.
    4. Save metrics and generations.
    """

    def __init__(self, config: BaselineConfig):
        self.config = config
        self.verifier = MathVerifier()
        self.llm: Optional[LLM] = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "generations"), exist_ok=True)

    def _init_llm(self):
        """Lazy initialization of vLLM engine."""
        if self.llm is None:
            print(f"Initializing vLLM with model: {self.config.model_path}")
            self.llm = LLM(
                model=self.config.model_path,
                tensor_parallel_size=self.config.tp_size,
                max_num_seqs=self.config.max_num_seqs,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                seed=self.config.seed,
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
        for item in dataset:
            prompt = item[self.config.prompt_key]
            if self.config.apply_chat_template:
                # Construct chat messages
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                # Note: vLLM's LLM class doesn't have a direct apply_chat_template like transformers
                # but we can use the tokenizer or assume a standard format for Qwen.
                # For Qwen2.5-Math-Instruct, the format is:
                # <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
                formatted_prompt = f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n" \
                                   f"<|im_start|>user\n{prompt}<|im_end|>\n" \
                                   f"<|im_start|>assistant\n"
                prompts.append(formatted_prompt)
            else:
                prompts.append(prompt)
        return prompts

    def generate_samples(self, prompts: List[str], temperature: float, n: int) -> List[Any]:
        """Generate N samples for each prompt using vLLM."""
        self._init_llm()
        
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )
        
        print(f"Generating {n} samples per prompt at temperature {temperature}...")
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
        
        return {k: float(np.mean(v)) for k, v in results.items()}

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

    def run(self):
        """Main execution flow."""
        dataset = self.load_dataset()
        prompts = self.prepare_prompts(dataset)
        ground_truths = [item[self.config.answer_key] for item in dataset]
        
        # Initialize WandB if requested
        if self.config.use_wandb:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config)
            )

        # Phase 1: Temperature Sweep
        print(f"\n--- Phase 1: Temperature Sweep (first {self.config.sweep_size} problems) ---")
        sweep_dataset = dataset[:self.config.sweep_size]
        sweep_prompts = prompts[:self.config.sweep_size]
        sweep_gts = ground_truths[:self.config.sweep_size]
        
        sweep_results = {}
        # We compute pass@1 and pass@N (where N is current config.n_samples)
        # TODO: User suggested N=8 for now, but we should consider larger N (e.g. 64, 128) in the future.
        metrics_to_track = [1, self.config.n_samples]
        
        best_temp = self.config.temperatures[0]
        best_val = -1.0
        
        for temp in self.config.temperatures:
            outputs = self.generate_samples(sweep_prompts, temp, self.config.n_samples)
            metrics = self.compute_metrics(outputs, sweep_gts, metrics_to_track)
            sweep_results[str(temp)] = metrics
            
            print(f"Temp {temp}: {metrics}")
            
            # Selector is pass@N
            current_val = metrics[f"pass@{self.config.n_samples}"]
            if current_val > best_val:
                best_val = current_val
                best_temp = temp
            
            if self.config.use_wandb:
                log_data = {f"sweep/temp_{temp}/{k}": v for k, v in metrics.items()}
                wandb.log(log_data)
        
        print(f"\nBest temperature selected: {best_temp} (pass@{self.config.n_samples} = {best_val:.4f})")
        self.save_results("sweep_results", {
            "best_temp": best_temp,
            "sweep_metrics": sweep_results
        })

        # Phase 2: Full Evaluation
        print(f"\n--- Phase 2: Full Evaluation (Temp={best_temp}) ---")
        full_outputs = self.generate_samples(prompts, best_temp, self.config.n_samples)
        
        # Compute final metrics (including intermediate k values for a nice curve)
        k_values = [1, 2, 4, 8] # Could add more if n_samples > 8
        if self.config.n_samples not in k_values:
            k_values.append(self.config.n_samples)
        k_values = sorted(list(set(k_values)))
        
        final_metrics = self.compute_metrics(full_outputs, ground_truths, k_values)
        print(f"Final Metrics: {final_metrics}")
        
        summary = {
            "config": asdict(self.config),
            "best_temp": best_temp,
            "final_metrics": final_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_results("summary", summary)
        
        if self.config.save_generations:
            self.save_generations(f"full_eval_temp_{best_temp}", full_outputs, dataset, best_temp)
            
        if self.config.use_wandb:
            wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Best-of-N Baseline for Math Reasoning")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--dataset_path", type=str, default="trex/data/gsm8k_platinum_test.jsonl")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--sweep_size", type=int, default=100)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="trex/results/bon_baseline")
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    config = BaselineConfig(
        model_path=args.model_path,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        n_samples=args.n_samples,
        sweep_size=args.sweep_size,
        tp_size=args.tp_size,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    baseline = BestOfNBaseline(config)
    baseline.run()


if __name__ == "__main__":
    main()
