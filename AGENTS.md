# Repository Guidelines

T-REX (Twisted Replica Exchange) is a research project implementing probabilistic inference methods for mathematical reasoning.

## Context Management

NOTE: Literally EVERY document in this repository except `docs/plans/trex_implementation_plan_20260317.md` and `docs/plans/trex_methodology_20260317.md` is stale.
The very first goal should be to update these documents before literally anything else.

- See `docs/HIGH_LEVEL_CONTEXT.md` for mathematical specification and algorithm details
- See `docs/plans/PROJECT_HIGH_LEVEL_PLAN.md` for details of the development roadmap with status. We will make more detailed plans for each item in this plan in separate Markdown files before any implementation is done.
- See `docs/EXPERIMENTS.md` to keep track of all experiments that have been done. This should contain all details about what experiments have been ran, what hypothesis we are testing with each experiment, and what the results are.
- Old docs from the Feb 2026 approach (Twisted SMC + Block-Gibbs transport) are archived in `docs/archive/2026-feb/`.

## Project Structure & Module Organization

- `openrlhf/`: Core RLHF framework (fork of OpenRLHF). CLI entry points live in `openrlhf/cli/`; model/trainer code in `openrlhf/models/` and `openrlhf/trainer/`.
- `trex/`: T-REX research code. Key areas include `trex/baselines/`, `trex/eval/`, `trex/smc/`, and `trex/utils/`.
- `trex/tests/`: pytest suite (directories prefixed with `test_`).
- `docs/`: documentation — `HIGH_LEVEL_CONTEXT.md` (math spec), `EXPERIMENTS.md`, `plans/` (implementation plans), `archive/2026-feb/` (old approach docs).
- `results/`, `slurm/`, `wandb/`: scratch symlinks for outputs and logs (HPC environment).

## Experiment Documentation

- After running any experiment, update `docs/EXPERIMENTS.md` immediately.
- Order of content: (1) results/metrics, (2) interpretation in the experiment context.
- Explicitly record: what command was run, whether anything went wrong (bugs, invalid runs, anomalies), and what the results imply for the project's next steps.
- Maintain a top-of-file "Current Experiment State (Summary)" section that lists the problems being solved, any solutions found, what is currently being tried, and a clearly labeled outstanding problem.
- The project pivoted in March 2026 from a Twisted SMC + Block-Gibbs transport approach to a simpler multi-chain SMC with mixture proposals. Old docs are archived in `docs/archive/2026-feb/`.

## Security & Configuration Tips

- It is possible that some compute nodes have no internet access. Set `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline` in SLURM scripts, and cache models ahead of time.
