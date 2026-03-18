# T-REX

T-REX is a research repository for inference-time reasoning with Sequential Monte Carlo on top of OpenRLHF.

The active design changed in March 2026. The source of truth is [docs/HIGH_LEVEL_CONTEXT.md](docs/HIGH_LEVEL_CONTEXT.md). If this README or any planning doc disagrees with that specification, follow `docs/HIGH_LEVEL_CONTEXT.md`.

The pre-pivot twisted/value-head/transport documents are archived under [docs/archive/2026-feb/](docs/archive/2026-feb/).

## Active Algorithm

The active T-REX formulation is a multi-chain SMC sampler over partial reasoning traces. It targets

`pi(x_{1:T}) propto P_theta(x_{1:T}) * R(x_{1:T})`

using a ladder of tempered chains

`pi_t^k(x_{1:t}) propto P_theta(x_{1:t}) * R(x_{1:t})^{beta_k}`

with `beta_1 = 0` and `beta_K = 1`.

Each chain runs the standard propagate -> reweight -> resample loop, with incremental weights

`w_t^{k,(i)} = (R(x_{1:t}^{k,(i)}) / R(x_{1:t-1}^{k,(i)}))^{beta_k}`.

Cold chains communicate with adjacent hot chains through mixture proposals: with probability `lambda`, a cold-chain particle is replaced by a full trajectory from the neighboring hotter chain and reweighted by `R(x_{1:t})^{Delta beta}`. This is the core contribution described in `docs/HIGH_LEVEL_CONTEXT.md`.

## Current Status

Active and reusable:

- Best-of-N and GRPO/PPO baselines
- Math verification, parsing, and grading utilities
- Reward-model wrapper for PRM/ORM-style scoring
- Resampling utilities and generic particle-filter infrastructure
- Stage 3 single-chain SMC core in `trex/smc/single_chain_smc.py`
- Stage 4 multi-chain SMC scaffold in `trex/smc/multi_chain_smc.py`

Still in progress:

- Wiring the active LLM + PRM stack to the new prefix-reward SMC formulation
- End-to-end Stage 3 validation against Best-of-N
- End-to-end Stage 4 validation and diagnostics

Archived:

- Twisted SMC
- Value-head / twist-model training path
- Transport-era TSMC runner and related scripts

## Repository Map

- `openrlhf/`: upstream RLHF framework used for training and distributed infrastructure
- `trex/baselines/`: BoN, GRPO, and related evaluation/training baselines
- `trex/eval/`: answer extraction, grading, and math verification
- `trex/models/`: reward-model wrappers and archived model shims
- `trex/smc/`: active SMC utilities and Stage 3/4 cores
- `trex/tests/`: pytest suite
- `docs/HIGH_LEVEL_CONTEXT.md`: mathematical specification for the active algorithm
- `docs/EXPERIMENTS.md`: experiment log and current experiment state
- `docs/archive/2026-feb/`: archived pre-pivot documents

## Installation

Basic editable install:

```bash
pip install -e .
```

Cluster-specific extras:

```bash
pip install -r requirements_tamia.txt
```

On compute nodes without internet, set:

```bash
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
```

## Running Tests

After `pip install -e .`:

```bash
pytest trex/tests/ -v
```

Fast unit-only subset:

```bash
pytest trex/tests/ -v -m "not integration"
```

## Running Experiments

Examples:

```bash
sbatch trex/scripts/tamia/run_grpo_baseline.sh
sbatch trex/scripts/tamia/run_sft_prm800k_h200.sh
```

Use [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) to track what was run, what failed, and what the results imply for the next iteration.

## Documentation

- Use [docs/HIGH_LEVEL_CONTEXT.md](docs/HIGH_LEVEL_CONTEXT.md) for the active algorithm and implementation stages.
- Use [docs/plans/PROJECT_HIGH_LEVEL_PLAN.md](docs/plans/PROJECT_HIGH_LEVEL_PLAN.md) for the execution plan derived from that spec.
- Use [docs/archive/2026-feb/](docs/archive/2026-feb/) only for historical context from the pre-pivot approach.
