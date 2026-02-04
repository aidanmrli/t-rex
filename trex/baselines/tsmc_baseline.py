"""
Twisted SMC baseline for math reasoning.

Uses a value head as twist (no PRM) and optional ORM for final selection.
"""

import argparse
import atexit
import json
import logging
import os
import re
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import jsonlines

from trex.baselines.tsmc_config import TSMCConfig, CheckpointManager
from trex.eval import MathVerifier

logger = logging.getLogger(__name__)


def _clean_response(text: str) -> str:
    """
    Strip chat template scaffolding and stop tokens from a prompt+response string.
    """
    text = re.sub(r"^.*?<\|im_start\|>assistant\n?", "", text, flags=re.DOTALL, count=1)
    for stop_word in ("</s>", "<|im_end|>", "<END_OF_TURN>"):
        if stop_word in text:
            text = text.split(stop_word)[0].strip()
    return text


class TSMCBaseline:
    """Twisted SMC baseline with SLURM checkpointing support."""

    def __init__(self, config: TSMCConfig):
        self.config = config
        self.verifier = MathVerifier()
        self.generator = None
        self.twist_model = None
        self.reward_model = None

        self._shutdown_requested = False
        self._checkpoint_saved_on_exit = False
        self._last_checkpoint_time = time.time()

        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "generations"), exist_ok=True)

        if config.enable_checkpointing:
            checkpoint_path = os.path.join(config.output_dir, config.checkpoint_file)
            self.checkpoint_mgr = CheckpointManager(
                checkpoint_path,
                config_hash=config.config_hash(),
            )
            self._setup_signal_handlers()
        else:
            self.checkpoint_mgr = None

    def _setup_signal_handlers(self) -> None:
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\nReceived {sig_name}, saving checkpoint and exiting gracefully...")
            self._shutdown_requested = True
            if self.checkpoint_mgr and not self._checkpoint_saved_on_exit:
                self._checkpoint_saved_on_exit = True
                self.checkpoint_mgr.save()
            sys.exit(0)

        def atexit_handler():
            if self.checkpoint_mgr and not self._checkpoint_saved_on_exit:
                self._checkpoint_saved_on_exit = True
                self.checkpoint_mgr.save()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
        atexit.register(atexit_handler)

    def _should_checkpoint(self, completed_idx: int) -> bool:
        """Check whether checkpoint should be saved."""
        if self.checkpoint_mgr is None:
            return False
        return self.checkpoint_mgr.should_save(
            idx=completed_idx,
            interval=self.config.checkpoint_interval,
            time_interval=self.config.checkpoint_time_interval,
            last_save_epoch=self._last_checkpoint_time,
        )

    def _save_checkpoint(self, completed_idx: int, results: List[Dict[str, Any]]) -> None:
        """Persist checkpoint state for resume support."""
        if self.checkpoint_mgr is None:
            return
        self.checkpoint_mgr.state["completed_idx"] = completed_idx
        self.checkpoint_mgr.state["results"] = results
        self.checkpoint_mgr.save()
        self._last_checkpoint_time = time.time()

    def _load_final_metrics(self) -> Dict[str, Any]:
        """Load previously computed metrics for finished runs."""
        metrics_path = os.path.join(self.config.output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {}

    def _init_generator(self) -> None:
        if self.generator is None:
            from vllm import LLM

            print(f"Initializing vLLM generator: {self.config.generator_model_path}")
            self.generator = LLM(
                model=self.config.generator_model_path,
                tensor_parallel_size=self.config.generator_tp_size,
                max_num_seqs=self.config.max_num_seqs,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                seed=self.config.seed,
            )

    def _init_twist_model(self) -> None:
        if self.twist_model is None:
            from trex.models.twist_model import TwistModel

            print(f"Initializing twist model: {self.config.value_model_path}")
            self.twist_model = TwistModel(
                model_name_or_path=self.config.value_model_path,
                value_head_type=self.config.value_head_type,
                twist_space=self.config.twist_space,
                freeze_base_model=True,
                share_base_with_generator=self.config.share_base_with_generator,
                epsilon=self.config.epsilon,
                log_value_min=self.config.log_value_min,
            )
            if self.config.value_head_path:
                import torch

                state = torch.load(self.config.value_head_path, map_location="cpu")
                self.twist_model.value_head.load_state_dict(state)

    def _init_reward_model(self) -> None:
        if not self.config.use_orm_for_final:
            return
        if self.reward_model is None:
            from trex.models.reward_model import RewardModel

            if self.config.reward_model_path is None:
                raise ValueError("reward_model_path is required when use_orm_for_final=True")

            print(f"Initializing ORM reward model: {self.config.reward_model_path}")
            self.reward_model = RewardModel(
                model_path=self.config.reward_model_path,
                tp_size=self.config.reward_model_tp_size,
            )

    def _init_models(self) -> None:
        self._init_generator()
        self._init_twist_model()
        self._init_reward_model()

    def load_dataset(self) -> List[Dict[str, Any]]:
        print(f"Loading dataset from {self.config.dataset_path}")
        dataset = []
        with jsonlines.open(self.config.dataset_path) as reader:
            for obj in reader:
                dataset.append(obj)
        print(f"Loaded {len(dataset)} problems")
        return dataset

    def prepare_prompt(self, problem: str) -> str:
        if not self.config.apply_chat_template:
            if self.config.resampling_unit == "token":
                return problem
            if self.config.step_boundary_mode == "delimiter":
                return problem
            return problem + "\n\n## Step 1:"

        self._init_generator()
        tokenizer = self.generator.get_tokenizer()
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": problem},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.config.resampling_unit == "token":
            return prompt
        if self.config.step_boundary_mode == "delimiter":
            return prompt
        return prompt + "\n\n## Step 1:"

    def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        from trex.smc.tsmc_particle_filter import TSMCLLMParticleFilter

        self._init_models()

        problem = item[self.config.prompt_key]
        ground_truth = item[self.config.answer_key]
        prompt = self.prepare_prompt(problem)

        pf = TSMCLLMParticleFilter(
            config=self.config,
            generator=self.generator,
            twist_scorer=self.twist_model,
            reward_model=self.reward_model,
        )

        pf.initialize(prompt)
        start_time = time.time()
        best_particle = pf.run()
        elapsed_time = time.time() - start_time

        final_text = best_particle.text
        response_text = _clean_response(final_text)
        extracted_answer = self.verifier.extract_answer(response_text)
        is_correct = self.verifier.verify(response_text, ground_truth)

        summary = pf.get_summary()

        return {
            "problem": problem,
            "ground_truth": ground_truth,
            "final_text": final_text,
            "final_response": response_text,
            "final_answer": extracted_answer,
            "correct": bool(is_correct),
            "smc_iteration": summary.get("smc_iteration"),
            "ess": summary.get("ess"),
            "elapsed_time": elapsed_time,
        }

    def run(self) -> Dict[str, Any]:
        if self.checkpoint_mgr and self.checkpoint_mgr.is_finished():
            print("Experiment already completed (found finished checkpoint). Exiting.")
            return self._load_final_metrics()

        dataset = self.load_dataset()
        results = []
        correct = 0

        resume_idx = 0
        if self.checkpoint_mgr:
            resume_idx = self.checkpoint_mgr.get_resume_index()
            if resume_idx > 0:
                results = self.checkpoint_mgr.get_results()
                correct = sum(1 for r in results if r.get("correct"))

        for idx in range(resume_idx, len(dataset)):
            if self._shutdown_requested:
                break
            item = dataset[idx]
            result = self.evaluate_single(item)
            results.append(result)
            correct += 1 if result["correct"] else 0

            if self._should_checkpoint(idx + 1):
                self._save_checkpoint(idx + 1, results)

        accuracy = correct / max(1, len(results))
        metrics = {
            "accuracy": accuracy,
            "n": len(results),
            "config": self.config.to_dict(),
        }

        out_path = os.path.join(self.config.output_dir, "metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if self.checkpoint_mgr:
            self.checkpoint_mgr.mark_finished()

        return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twisted SMC baseline runner.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--generator_model_path", type=str, default=None)
    parser.add_argument("--value_model_path", type=str, default=None)
    parser.add_argument("--value_head_path", type=str, default=None)
    parser.add_argument("--value_head_type", type=str, default=None)
    parser.add_argument("--twist_space", type=str, default=None)
    parser.add_argument("--twist_mode", type=str, default=None)
    parser.add_argument("--n_particles", type=int, default=None)
    parser.add_argument("--max_smc_iterations", type=int, default=None)
    parser.add_argument("--resampling_unit", type=str, default=None)
    parser.add_argument("--resample_every_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--step_boundary_mode", type=str, default=None)
    parser.add_argument("--step_delimiter", type=str, default=None)
    return parser


def load_config(args: argparse.Namespace) -> TSMCConfig:
    if args.config:
        with open(args.config, "r") as f:
            data = json.load(f)
        cfg = TSMCConfig.from_dict(data)
    else:
        cfg = TSMCConfig()

    overrides = {
        "output_dir": args.output_dir,
        "dataset_path": args.dataset_path,
        "generator_model_path": args.generator_model_path,
        "value_model_path": args.value_model_path,
        "value_head_path": args.value_head_path,
        "value_head_type": args.value_head_type,
        "twist_space": args.twist_space,
        "twist_mode": args.twist_mode,
        "n_particles": args.n_particles,
        "max_smc_iterations": args.max_smc_iterations,
        "resampling_unit": args.resampling_unit,
        "resample_every_tokens": args.resample_every_tokens,
        "temperature": args.temperature,
        "step_boundary_mode": args.step_boundary_mode,
        "step_delimiter": args.step_delimiter,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    # Reconstruct to ensure __post_init__ validation runs after CLI overrides.
    return TSMCConfig.from_dict(cfg.to_dict())


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args)
    baseline = TSMCBaseline(config)
    metrics = baseline.run()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
