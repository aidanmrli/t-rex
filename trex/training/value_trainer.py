"""
Value head training via self-distillation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import random
import re

import torch
from torch import nn
from torch.optim import AdamW

from trex.training.trajectory_buffer import Trajectory, TrajectoryBuffer


@dataclass
class ValueTrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_rollouts_per_prompt: int = 8
    update_frequency: int = 100
    ema_decay: float = 0.99
    max_steps: int = 1000
    max_seq_len: Optional[int] = None
    max_tokens_per_state: Optional[int] = None
    class_balance_target: Optional[float] = None
    label_smoothing: float = 0.0
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    step_pattern: str = r"## Step \d+:"
    step_boundary_mode: str = "header"
    step_delimiter: str = "\n\n"
    temperature: float = 0.8
    max_tokens: int = 2048
    apply_chat_template: bool = True
    system_prompt: str = (
        "Solve the following math problem efficiently and clearly:\n"
        "- For simple problems (2 steps or fewer): Provide a concise solution with minimal explanation.\n"
        "- For complex problems (3 steps or more): Use this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n"
        "[Brief explanation and calculations]\n\n"
        "Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem."
    )


class ValueTrainer:
    """Trainer for value head via self-distillation."""

    def __init__(self, twist_model, config: ValueTrainingConfig):
        self.twist_model = twist_model
        self.config = config
        self.optimizer = AdamW(twist_model.value_head.parameters(), lr=config.learning_rate)
        self.buffer = TrajectoryBuffer()
        self._step_re = re.compile(config.step_pattern)

    def _split_steps_by_delimiter(self, text: str) -> List[str]:
        delimiter = self.config.step_delimiter
        if not delimiter:
            return [text] if text.strip() else []
        if delimiter not in text:
            return [text] if text.strip() else []

        parts = text.split(delimiter)
        steps: List[str] = []
        for idx, part in enumerate(parts):
            if idx < len(parts) - 1:
                steps.append(part + delimiter)
            elif part.strip():
                steps.append(part)
        return steps

    def _split_steps(self, text: str) -> List[str]:
        if self.config.step_boundary_mode == "delimiter":
            return self._split_steps_by_delimiter(text)

        matches = list(self._step_re.finditer(text))
        if not matches:
            return [text] if text.strip() else []
        steps = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            steps.append(text[start:end])
        return steps

    def _prepare_prompt(self, problem: str, generator) -> str:
        if not self.config.apply_chat_template:
            return problem
        tokenizer = generator.get_tokenizer()
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": problem},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def collect_rollouts(
        self,
        problems: List[Dict],
        generator,
        verifier,
    ) -> List[Trajectory]:
        """Generate trajectories and assign rewards."""
        from vllm import SamplingParams

        trajectories: List[Trajectory] = []
        if not problems:
            return trajectories

        raw_prompts = [p[self.config.prompt_key] for p in problems]
        prompts = [self._prepare_prompt(p, generator) for p in raw_prompts]
        answers = [p[self.config.answer_key] for p in problems]

        sampling_params = SamplingParams(
            n=self.config.num_rollouts_per_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=None,
            include_stop_str_in_output=False,
        )

        outputs = generator.generate(prompts, sampling_params)

        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            ground_truth = answers[idx]
            for candidate in output.outputs:
                text = prompt + candidate.text
                extracted = verifier.extract_answer(text)
                reward = 1.0 if verifier.verify(extracted, ground_truth) else 0.0
                # Split steps from generated continuation so prompt is not duplicated
                # when converting trajectories to (state, reward) pairs.
                steps = self._split_steps(candidate.text)
                trajectories.append(
                    Trajectory(
                        prompt=prompt,
                        steps=steps,
                        full_text=text,
                        reward=reward,
                    )
                )

        return trajectories

    def _prepare_batch(self, pairs: List[Tuple[str, float]]) -> Tuple[List[str], torch.Tensor]:
        if not pairs:
            return [], torch.tensor([])

        if self.config.class_balance_target is None:
            batch = random.sample(pairs, min(self.config.batch_size, len(pairs)))
        else:
            positives = [p for p in pairs if p[1] > 0.5]
            negatives = [p for p in pairs if p[1] <= 0.5]
            target_pos = int(self.config.batch_size * self.config.class_balance_target)
            target_neg = self.config.batch_size - target_pos

            pos_sample = random.sample(positives, min(len(positives), target_pos)) if positives else []
            neg_sample = random.sample(negatives, min(len(negatives), target_neg)) if negatives else []

            batch = pos_sample + neg_sample
            if len(batch) < self.config.batch_size:
                remainder = [p for p in pairs if p not in batch]
                if remainder:
                    batch += random.sample(remainder, min(len(remainder), self.config.batch_size - len(batch)))
        states, rewards = zip(*batch)
        targets = torch.tensor(rewards, dtype=torch.float32)

        if self.config.label_smoothing > 0.0:
            smoothing = self.config.label_smoothing
            targets = targets * (1.0 - smoothing) + (1.0 - targets) * smoothing

        return list(states), targets

    def train_step(self, trajectories: List[Trajectory]) -> float:
        """Single training step. Returns loss."""
        pairs: List[Tuple[str, float]] = []
        for traj in trajectories:
            pairs.extend(traj.get_state_reward_pairs())

        if not pairs:
            return 0.0

        states, targets = self._prepare_batch(pairs)
        if not states:
            return 0.0

        logits = self.twist_model.score_texts_logits(states)
        logits = logits.view(-1).to(targets.device)

        pos_weight = None
        if self.config.class_balance_target is not None:
            pos = targets.sum().item()
            neg = max(1.0, float(len(targets) - pos))
            if pos > 0:
                pos_weight = torch.tensor([neg / pos], device=targets.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(logits, targets.to(logits.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train(self, dataset: List[Dict], verifier, generator) -> None:
        """Full training loop with periodic rollout collection."""
        step = 0
        while step < self.config.max_steps:
            if step % self.config.update_frequency == 0 or not self.buffer.trajectories:
                batch = random.sample(dataset, min(len(dataset), self.config.batch_size))
                trajectories = self.collect_rollouts(batch, generator, verifier)
                for traj in trajectories:
                    self.buffer.add(traj)

            sample_trajectories = self.buffer.sample(self.config.batch_size)
            loss = self.train_step(sample_trajectories)
            step += 1

            if step % 50 == 0:
                print(f"[ValueTrainer] step={step} loss={loss:.4f}")
