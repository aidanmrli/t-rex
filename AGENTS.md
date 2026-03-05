# Repository Guidelines

## Project Structure & Module Organization
- `openrlhf/`: Core RLHF framework (fork of OpenRLHF). CLI entry points live in `openrlhf/cli/`; model/trainer code in `openrlhf/models/` and `openrlhf/trainer/`.
- `trex/`: T-REX research code. Key areas include `trex/baselines/`, `trex/eval/`, `trex/smc/`, and `trex/utils/`.
- `trex/tests/`: pytest suite (directories prefixed with `test_`).
- `docs/`: documentation — `HIGH_LEVEL_CONTEXT.md` (math spec), `EXPERIMENTS.md`, `plans/` (implementation plans), `archive/2026-feb/` (old approach docs).
- `results/`, `slurm/`, `wandb/`: scratch symlinks for outputs and logs (HPC environment).

## Build, Test, and Development Commands
- Install (editable): `pip install -e .`
- Cluster/extra deps: `pip install -r requirements_tamia.txt` (see `CLAUDE.md` for module setup).
- Run tests: `pytest trex/tests/ -v`
- Coverage: `pytest trex/tests/ --cov=trex --cov-report=html`
- SLURM jobs: `sbatch trex/scripts/tamia/run_grpo_baseline.sh`

## Coding Style & Naming Conventions
- Python, 4-space indentation.
- Line length: 119 (Black/isort/Ruff configured in `pyproject.toml`).
- Formatters: Black + isort; cleanup via autoflake. Prefer `pre-commit run --all-files` before PRs.
- Tests and modules use `snake_case`; test files and folders use `test_*`.

## Testing Guidelines
- Framework: pytest with markers (`unit`, `integration`, `system`, `acceptance`, `docs`).
- Example marker usage: `pytest trex/tests/ -m "not integration" -v`.
- Keep tests isolated; add new tests alongside the subsystem in `trex/tests/test_*`.

## Experiment Documentation
- After running any experiment, update `docs/EXPERIMENTS.md` immediately.
- Order of content: (1) results/metrics, (2) interpretation in the experiment context.
- Explicitly record: what command was run, whether anything went wrong (bugs, invalid runs, anomalies), and what the results imply for the project's next steps.
- Maintain a top-of-file "Current Experiment State (Summary)" section that lists the problems being solved, any solutions found, what is currently being tried, and a clearly labeled outstanding problem.
- The project pivoted in March 2026 from a Twisted SMC + Block-Gibbs transport approach to a simpler multi-chain SMC with mixture proposals. Old docs are archived in `docs/archive/2026-feb/`.

## Commit & Pull Request Guidelines
- Follow observed conventions: short imperative subjects, often Conventional Commits like `feat:`, `fix:`, `docs:`, with optional scopes (`fix(smc): ...`).
- PRs should include: brief summary, tests run, linked issue (if any), and before/after notes for metric or eval changes. Include SLURM job details when relevant.

## Security & Configuration Tips
- Compute nodes have no internet access. Set `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline` in SLURM scripts, and cache models in `/scratch/l/liaidan/model_weights` ahead of time.
