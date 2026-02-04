"""
CLI entry point for value head training.
"""

import argparse
import json
from typing import Any, Dict, List

import jsonlines

from trex.eval import MathVerifier
from trex.models.twist_model import TwistModel
from trex.training.value_trainer import ValueTrainer, ValueTrainingConfig


def load_dataset(path: str) -> List[Dict[str, Any]]:
    dataset = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            dataset.append(obj)
    return dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train value head via self-distillation.")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--value_head_type", type=str, default="mlp")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--update_frequency", type=int, default=100)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--twist_space", type=str, default="log_prob")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--no_chat_template", action="store_true", help="Disable chat template formatting.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = load_dataset(args.dataset)
    verifier = MathVerifier()

    from vllm import LLM

    generator = LLM(
        model=args.base_model,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    twist_model = TwistModel(
        model_name_or_path=args.base_model,
        value_head_type=args.value_head_type,
        twist_space=args.twist_space,
        freeze_base_model=True,
    )

    config = ValueTrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_rollouts_per_prompt=args.num_rollouts_per_prompt,
        update_frequency=args.update_frequency,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        apply_chat_template=not args.no_chat_template,
    )

    trainer = ValueTrainer(twist_model, config)
    trainer.train(dataset, verifier=verifier, generator=generator)

    # Save value head weights
    import os
    import torch

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "value_head.pt")
    torch.save(twist_model.value_head.state_dict(), out_path)

    meta = {
        "base_model": args.base_model,
        "value_head_type": args.value_head_type,
        "twist_space": args.twist_space,
        "config": config.__dict__,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
