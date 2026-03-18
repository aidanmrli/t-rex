# Project High-Level Plan for T-REX

**Last Updated:** 2026-03-05

`docs/HIGH_LEVEL_CONTEXT.md` is the source of truth for the active algorithm. This file is an execution plan derived from that spec. If the two disagree, follow `docs/HIGH_LEVEL_CONTEXT.md`.

Archived pre-pivot material lives under `docs/archive/2026-feb/`.

---

## Goal

Implement and validate the March 2026 T-REX formulation:

- single-chain SMC over partial reasoning traces
- a temperature ladder with `beta_1 = 0 < ... < beta_K = 1`
- hot-to-cold mixture proposals with full-trajectory injection
- diagnostics and experiments showing the method improves over Best-of-N and single-chain SMC

The target posterior is

`pi(x_{1:T}) propto P_theta(x_{1:T}) * R(x_{1:T})`

with tempered intermediate targets

`pi_t^k(x_{1:t}) propto P_theta(x_{1:t}) * R(x_{1:t})^{beta_k}`.

The key incremental weight for local propagation is

`w_t^{k,(i)} = (R(x_{1:t}^{k,(i)}) / R(x_{1:t-1}^{k,(i)}))^{beta_k}`.

The key injected-particle weight is

`w_{t,inject}^{(i)} = R(x_{1:t}^{(i)})^{beta_{k+1} - beta_k}`.

---

## Current Repository State

Reusable today:

- `trex/eval/`: answer parsing, grading, verification
- `trex/models/reward_model.py`: PRM/ORM wrapper surface
- `trex/smc/resampling.py`: ESS and resampling utilities
- `trex/smc/particle_filter.py`: generic particle container utilities
- `trex/smc/llm_particle_filter.py`: generation/prompt-prep scaffolding that may be partially reused
- `trex/smc/single_chain_smc.py`: Stage 3 math core
- `trex/smc/multi_chain_smc.py`: Stage 4 scaffold
- `trex/baselines/`: BoN and RL baselines for comparison

Archived or not for new development:

- twisted SMC
- value-head / twist-model path
- transport-era TSMC baseline
- old tempering / swap-based design documents

Still missing or incomplete:

- prefix-level PRM scoring contract for the new SMC math
- end-to-end Stage 3 integration with the active LLM path
- fully correct and validated Stage 4 integration
- Stage 5 diagnostics and Stage 7 evaluation for the new algorithm

---

## Workstreams

### 1. Cleanup and Interface Hardening

Objective:
- keep active docs and package surfaces aligned with the new algorithm
- move dead documents and stale entry points out of the active path

Tasks:
- archive stale docs that still describe twisted/value-head/transport-era T-REX
- remove or clearly disable old runtime entry points that still suggest the archived stack is usable
- keep active docs limited to the multi-chain SMC formulation

Exit criteria:
- active docs all point to `docs/HIGH_LEVEL_CONTEXT.md`
- archived paths are clearly marked historical
- new contributors can find the active algorithm without touching the old stack

### 2. Base LLM and PRM Interface

Objective:
- expose a clean runtime interface for the Stage 3/4 sampler

Tasks:
- define batched next-token sampling from the base model
- define prefix-level PRM scoring for partial trajectories
- ensure PRM scores are strictly positive and cached per particle
- settle one numerical stack boundary between Torch and NumPy to avoid unnecessary conversions

Exit criteria:
- we can evaluate `R(x_{1:t})` for every active particle at every step
- we can reuse the same interface for both single-chain and multi-chain execution

### 3. Stage 3: Single-Chain SMC

Objective:
- get one chain working end to end before touching multi-chain experiments

Implementation requirements:
- propagate -> reweight -> resample loop
- log-space incremental weights using `beta * (log R_curr - log R_prev)`
- ESS-triggered systematic resampling
- cached per-particle trajectories and prefix rewards
- log normalizing-constant tracking
- EOS handling that freezes completed particles correctly

Validation requirements:
- `beta = 0` behaves like pure base-model sampling
- `beta = 1` concentrates on higher-reward trajectories
- finite `log Z`
- single-chain SMC at `beta = 1` is compared against BoN at equal `N`

Current status:
- the math core exists in `trex/smc/single_chain_smc.py`
- the active LLM/PRM path still needs to be wired to it

### 4. Stage 4: Multi-Chain SMC with Mixture Proposals

Objective:
- implement the actual T-REX contribution once Stage 3 is stable

Implementation requirements:
- `K` chains with increasing `beta`
- Bernoulli coin flip per particle for local extension vs hot injection
- full-trajectory replacement on injection
- correct injected-particle weighting under the proposal actually used
- per-chain ESS, resampling, ancestor tracking, and injection diagnostics

Validation requirements:
- injected particles survive at a non-trivial rate in colder chains
- cold-chain diversity exceeds single-chain SMC
- cold-chain accuracy meets or exceeds single-chain SMC at `beta = 1`
- scaling in `N` improves behavior rather than destabilizing it

Current status:
- the core scaffold exists in `trex/smc/multi_chain_smc.py`
- correctness and end-to-end validation remain incomplete

### 5. Diagnostics and Evaluation

Objective:
- make failure modes visible and produce the comparisons needed to justify the method

Diagnostics:
- per-chain ESS over time
- per-chain `log Z`
- injection attempt and survival rate
- particle genealogy
- reward distributions by chain and step

Experiments:
- single-chain SMC vs Best-of-N at equal `N`
- multi-chain T-REX vs single-chain SMC
- `lambda = 0` ablation
- temperature-schedule sweeps

Exit criteria:
- every major claim in `docs/HIGH_LEVEL_CONTEXT.md` has a concrete experiment attached to it

---

## Immediate Task Order

1. Finish cleanup of stale docs and dead entry points.
2. Define the new prefix-reward interface for the active LLM + PRM path.
3. Wire `trex/smc/single_chain_smc.py` into an end-to-end sampler and validate `beta=0` / `beta=1`.
4. Correct and validate `trex/smc/multi_chain_smc.py`.
5. Add diagnostics needed for debugging and reporting.
6. Run the comparison suite against BoN and single-chain SMC.

---

## Success Criteria

We are done with the pivot when all of the following are true:

- the active docs and package surfaces consistently reflect the multi-chain SMC design
- Stage 3 runs end to end with prefix-reward reweighting and passes the core sanity checks
- Stage 4 runs end to end with correct injection semantics and measurable hot-to-cold communication
- the experiment log contains BoN, single-chain SMC, multi-chain T-REX, and `lambda=0` comparisons
- the archived twisted/value-head/transport material is no longer part of the active development path
