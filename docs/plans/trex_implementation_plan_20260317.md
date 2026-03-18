
# T-REX Implementation Plan and Research Agenda

## Purpose

This document is the execution plan for building the repository implementation of T-REX. It is intentionally detailed and staged. It separates:

- what must be built,
- what can be deferred,
- what remains underspecified in the methodology,
- what must be researched or experimentally validated,
- and what should count as a successful milestone.

The recommended implementation strategy is to build the **core online sampler first** and defer amortized proposal learning until the online system is stable and measurable.

---

## 1. Project goals

### 1.1 Primary goal

Implement a production-quality blockwise tempered SMC system for reasoning trajectories that can:

- sample reasoning prefixes block by block,
- score prefixes with a process reward model,
- maintain weighted particle systems,
- perform resampling based on ESS,
- communicate across temperature chains through hot-bank injection,
- and return completed cold-chain trajectories for evaluation.

### 1.2 Secondary goal

Build the system so it can later support:

- amortized twisted proposals,
- temperature-conditioned proposal learning,
- richer reward models,
- optional editing or infill proposals,
- larger distributed evaluations.

### 1.3 Non-goals for the first implementation

The following should not be treated as first-pass requirements:

- online twist learning during inference,
- auxiliary masking / infill editing networks,
- classical swap-based PT as the primary method,
- full distributed multi-node optimization before correctness is established,
- broad benchmark coverage before core algorithm validation.

---

## 2. Design principles

1. **Faithfulness before optimization**  
   Implement exact branch-aware importance weighting before making performance shortcuts.

2. **Blockwise state is foundational**  
   Blockization is not a utility; it defines the state space. Treat it as a core module.

3. **The PRM is a central dependency**  
   The sampler’s behavior is dominated by reward quality and reward calibration.

4. **Separate “must implement” from “must research”**  
   The proposal leaves several high-impact design variables open. Capture them explicitly rather than hard-coding assumptions.

5. **Preserve observability**  
   Every stage should emit metrics, traces, particle ancestry, and diagnostic summaries.

---

## 3. Repository architecture recommendation

Recommended top-level modules:

```text
trex/
  config/
  blockization/
  models/
    base_model/
    prm/
    optional_twist/
  sampling/
    smc/
    tempered/
    resampling/
    hot_bank/
  inference/
  training/
    prm/
    optional_twist/
  evaluation/
  analysis/
  logging/
  utils/
docs/
experiments/
tests/
```

Recommended documentation files in the repo:

- `docs/methodology.md`
- `docs/implementation_plan.md`
- `docs/blockization.md`
- `docs/prm.md`
- `docs/metrics_and_logging.md`
- `docs/evaluation_protocol.md`

---

## 4. Stage 0: Formalization and interface design

## 4.1 Deliverables

Before writing most runtime code, define stable interfaces for:

- block generation,
- block boundary detection,
- prefix representation,
- PRM scoring,
- particle objects,
- chain objects,
- bank objects,
- resampling API,
- logging and tracing,
- inference outputs.

## 4.2 Required types and abstractions

### Prefix object
Should minimally contain:

- token ids,
- decoded text,
- block list,
- current depth,
- terminal flag,
- optional final-answer extraction,
- optional cached model state pointer,
- reward history,
- provenance metadata.

### Particle object
Should minimally contain:

- prefix,
- current log weight,
- parent particle id,
- chain id,
- branch type (`loc` or `hot`),
- step index,
- resample ancestry,
- terminal status.

### Chain object
Should minimally contain:

- inverse temperature \(\beta_k\),
- particle set,
- normalized weights,
- ESS,
- current bank,
- chain-level metrics.

### Bank object
Should minimally contain:

- depth index,
- chain id,
- prefix handles or materialized prefixes,
- normalized weights,
- optional deduplicated key,
- sampling API.

## 4.3 Research questions for Stage 0

These should be documented before implementation, even if they are unresolved.

1. What constitutes prefix identity for bank storage?
   - raw token sequence,
   - canonicalized text,
   - block list,
   - or hash of a normalized representation.

2. What is the exact semantics of a completed block?
   - newline-delimited step,
   - special delimiter,
   - parser-detected semantic unit,
   - model-emitted boundary token.

3. How are terminal trajectories represented?
   - explicit EOS block,
   - separate terminal flag,
   - both.

4. How should per-particle KV caches be handled?
   - copied,
   - shared through prefix trees,
   - lazily rehydrated.

---

## 5. Stage 1: Plain blockwise SMC with one chain

This is the minimum scientifically meaningful implementation.

## 5.1 Goal

Implement blockwise SMC with:

- one chain,
- one temperature (\(\beta = 1\)),
- base-model propagation,
- PRM-based incremental weighting,
- ESS-triggered resampling,
- absorbing-state handling.

## 5.2 Why this stage matters

Without a strong single-chain baseline, multi-temperature communication cannot be debugged. This stage isolates:

- blockization,
- PRM integration,
- weight stability,
- resampling correctness,
- and runtime cost.

## 5.3 Components to build

### 5.3.1 Blockizer v1
A first-pass blockization strategy is needed even if imperfect.

Recommended initial design:

- choose one explicit block delimiter policy,
- make it deterministic,
- make it reversible,
- and expose instrumentation for boundary diagnostics.

Suggested v1 options to evaluate:

1. newline-terminated reasoning steps,
2. delimiter-token terminated blocks,
3. parser-assisted equation or code block boundaries.

For v1, prefer the simplest scheme that is deterministic and debuggable.

### 5.3.2 Base-model block sampler
Implement a function:

```python
sample_next_block(prefix, sampling_config) -> BlockSample
```

It should return:

- block text,
- block token ids,
- token-level logprobs if available,
- stop reason,
- optional updated cache handle.

### 5.3.3 Prefix reward model adapter
Implement a PRM interface that can score a batch of completed prefixes:

```python
score_prefixes(prefixes) -> log_rewards
```

The interface should work in log-space by default.

### 5.3.4 SMC engine
Implement:

- propagation,
- incremental log-weight update,
- ESS computation,
- multinomial or systematic resampling,
- absorbing-state handling.

## 5.4 Metrics to collect

At every depth:

- ESS,
- normalized entropy of weights,
- number of active versus terminal particles,
- reward increment statistics,
- duplicate-prefix rate,
- block length statistics,
- PRM latency,
- sampler latency.

## 5.5 Tests required

### Unit tests
- reward-ratio update identity,
- ESS computation,
- resampling normalization,
- terminal-state persistence,
- blockizer determinism.

### Property tests
- normalized weights sum to one,
- resampled particle count is preserved,
- terminal particles do not mutate after absorption.

### Integration tests
- one short prompt with deterministic sampling seed,
- one prompt with forced early EOS,
- one prompt with long reasoning trajectory.

## 5.6 Research questions in Stage 1

1. **Which blockization policy is stable enough for sequential weighting?**  
   This is the biggest unresolved issue.

2. **How noisy are reward increments?**  
   Measure the distribution of \(\Delta \ell_m\).

3. **How aggressively should increments be clipped or normalized?**  
   This may be necessary for stability but changes effective behavior.

4. **What resampling threshold works best?**  
   The methodology leaves ESS threshold open.

5. **How should final outputs be selected?**  
   Options:
   - posterior sample,
   - max-weight completed trajectory,
   - best verifier score,
   - best PRM score among completed trajectories.

This choice should be explicitly documented and benchmarked.

## 5.7 Exit criteria

Stage 1 is complete when:

- the one-chain SMC implementation is correct and reproducible,
- end-to-end runs complete on a small benchmark subset,
- logs show stable weight updates,
- and debugging traces can explain particle survival or collapse.

---

## 6. Stage 2: Calibrated PRM and reward diagnostics

The methodology makes the PRM a first-class component. It deserves its own stage rather than being treated as an interchangeable scorer.

## 6.1 Goal

Improve the PRM so it is usable for sequential weighting rather than only ranking.

## 6.2 Components to build

### 6.2.1 PRM data pipeline
Construct training examples from:

- ancestral samples from the base model,
- teacher-forced or curated demonstrations,
- trajectories discovered by search or SMC.

Each example must preserve block boundaries consistent with inference.

### 6.2.2 Labeling pipeline
Support multiple label targets:

- binary eventual correctness,
- soft success probability,
- pairwise prefix preference,
- Monte Carlo reward-to-go.

Recommended initial target:
**estimated probability that the current completed prefix can still reach a correct final solution.**

### 6.2.3 Calibration pipeline
Implement:

- temperature scaling,
- isotonic calibration,
- block-depth stratified calibration,
- optional increment regularization.

### 6.2.4 Reward stabilization utilities
Implement utilities for:

- log-reward clipping,
- increment clipping,
- z-normalization per depth,
- optional moving-average diagnostics.

## 6.3 Experiments to run

1. Ranking-only PRM vs calibrated sequential PRM.
2. Raw reward vs clipped reward increments.
3. Prefix-length-stratified calibration vs global calibration.
4. Boundary-sensitive blockization variants and their effect on PRM stability.

## 6.4 Open research questions

1. What target best matches the sequential objective:
   local step quality, future correctness, or reward-to-go?
2. Does the PRM need to explicitly condition on block depth?
3. How much calibration drift occurs when blockization changes?
4. Should \(R\) be trained as a probability surrogate or only as an order-preserving potential?
5. What lower-bound floor on reward values is needed to preserve non-vanishing support without over-flattening the target?

## 6.5 Exit criteria

Stage 2 is complete when:

- PRM scores are stable across block depths,
- reward increments do not exhibit pathological variance,
- and single-chain SMC improves over naïve ancestral sampling under controlled compute.

---

## 7. Stage 3: Multi-chain tempering without communication

Before adding hot-bank injection, verify the per-chain temperature behavior.

## 7.1 Goal

Run multiple chains with different inverse temperatures but no cross-chain communication.

## 7.2 Why this stage matters

It isolates the effect of reward tempering from the effect of hot-bank transfer.

## 7.3 Components to build

- temperature ladder configuration,
- independent chain scheduling,
- per-chain metrics collection,
- temperature-aware reward exponentiation.

## 7.4 Core experiments

1. Compare chain behavior at \(\beta=0\), intermediate \(\beta\), and \(\beta=1\).
2. Measure:
   - prefix diversity,
   - reward statistics,
   - completion rate,
   - final correctness,
   - overlap between chains.

## 7.5 Questions to research

1. How many chains are necessary?
2. How should temperatures be spaced?
   - linear in \(\beta\),
   - geometric in effective reward scale,
   - adaptive based on chain overlap.
3. Does \(\beta=0\) alone provide sufficiently diverse exploration?
4. Should token sampling temperature also vary with chain, or should only reward tempering vary?

This last question is important because the proposal mathematically tempers reward, not decoding temperature. The repository should keep those concepts separate.

## 7.6 Exit criteria

Stage 3 is complete when:

- temperature-only behavior is understood,
- and there is evidence that different chains occupy meaningfully different reasoning regimes.

---

## 8. Stage 4: Hot-bank communication

This stage implements the core novelty of the proposal.

## 8.1 Goal

Add hot-bank resampling from hotter chains into colder chains using exact branch-aware importance weighting.

## 8.2 Components to build

### 8.2.1 Bank storage and sampling
Implement bank refresh and sampling APIs:

```python
bank.sample(weighted=True) -> PrefixRef
```

Support:

- weighted categorical sampling,
- optional deduplication,
- optional capped bank size,
- lineage metadata.

### 8.2.2 Branch sampler
Implement branch indicator sampling with configurable \(\lambda\).

### 8.2.3 Branch-specific weighting
Implement exact local and hot branch corrections under the augmented-state view.

### 8.2.4 Logging and debugging
Record for every particle update:

- branch used,
- parent source chain,
- parent particle id,
- prefix copied or extended,
- pre- and post-update log weight,
- whether the resulting particle survived resampling.

## 8.3 Crucial correctness checks

1. Local and hot branches must both preserve valid normalized weights.
2. The code must never require arbitrary empirical prefix lookup outside bank contents.
3. Hot-branch particles must preserve copied prefix identity exactly.
4. Communication must happen at aligned block depths.

## 8.4 Experiments to run

1. Vary \(\lambda\) from 0 to 1.
2. Compare:
   - no communication,
   - communication only between adjacent temperatures,
   - more aggressive communication policies.
3. Measure:
   - cold-chain diversity,
   - correctness,
   - ESS stability,
   - bank reuse,
   - duplicate injection rate.

## 8.5 Major research questions

1. What value of \(\lambda\) provides useful transfer without collapsing diversity?
2. Should bank sampling allow duplicate entries or deduplicate before sampling?
3. Should communication be strictly adjacent-chain or allow skipping temperatures?
4. At what cadence should banks refresh?
   - every block,
   - periodic,
   - event-triggered on ESS or diversity.
5. Should bank weights be smoothed before sampling to avoid over-concentrated injection?

## 8.6 Failure modes to monitor

- hot-chain prefixes dominate and erase cold-chain exploration,
- copied prefixes receive extreme correction weights,
- repeated hot injections create many duplicates,
- communication cost outweighs benefit,
- chain synchronization becomes brittle.

## 8.7 Exit criteria

Stage 4 is complete when:

- multi-chain communication runs correctly,
- branch-aware weighting is validated,
- and hot-bank communication outperforms a no-communication multi-chain baseline on at least one controlled benchmark slice.

---

## 9. Stage 5: Evaluation harness and baseline suite

After the core sampler works, build the evaluation framework.

## 9.1 Goal

Provide reproducible comparison against baselines under equal compute budgets.

## 9.2 Baselines to implement or wrap

Must-have baselines:

1. ancestral / best-of-N sampling,
2. beam or heuristic tree search,
3. standard one-chain SMC,
4. multi-chain no-communication tempering,
5. T-REX with hot-bank communication.

Nice-to-have later baselines:

6. MCTS with the same PRM,
7. twist-guided SMC,
8. swap-based PT or NRPT for direct comparison.

## 9.3 Benchmark sequencing

Recommended order:

### Phase A: math reasoning
- small GSM8K slice,
- small MATH/MATH500 slice.

### Phase B: code reasoning
- small HumanEval/EvalPlus slice.

### Phase C: harder or more exotic domains
- GPQA,
- APPS,
- SWE-bench Verified,
- constrained generation tasks.

Do not start broad. Start with one domain where debugging signals are interpretable.

## 9.4 Evaluation protocol

Every experiment should define:

- base model,
- blockization policy,
- PRM version,
- number of particles,
- number of chains,
- temperature ladder,
- communication probability \(\lambda\),
- token decoding configuration,
- compute budget,
- output selection rule.

## 9.5 Metrics

Primary metrics:

- pass@1 / accuracy / verifier correctness,
- compute-normalized accuracy,
- latency,
- tokens generated,
- PRM calls,
- wall-clock time.

Secondary metrics:

- unique completed trajectories,
- chain overlap,
- ESS curves,
- duplicate rate,
- bank utilization,
- reward increment variance.

---

## 10. Stage 6: Optional amortized proposal learning

Only start this stage after the online teacher is strong enough to generate useful training data.

## 10.1 Goal

Train a learned proposal or twist network using trajectories generated by the online tempered particle system.

## 10.2 Components to build

- offline teacher data exporter,
- weighted trajectory dataset format,
- twist parameterization,
- optional temperature-conditioned proposal model,
- forward-KL training objective,
- inference integration path.

## 10.3 Experimental priorities

1. Train on cold-chain posterior samples only.
2. Train on multi-temperature teacher data.
3. Compare:
   - unconditioned proposal,
   - \(\beta\)-conditioned proposal,
   - online teacher only,
   - learned proposal only,
   - learned proposal plus test-time SMC correction.

## 10.4 Open research questions

1. What proposal parameterization is tractable at the block level?
2. How should the intractable twist normalizer be approximated in practice?
3. Is it better to predict a twist potential, a direct proposal, or a small candidate reranker?
4. How much does forward-KL mass-covering behavior help compared with more mode-seeking objectives?
5. How much online correction is still needed after amortization?

## 10.5 Exit criteria

Stage 6 is complete when the learned proposal provides measurable speed/quality tradeoffs without invalidating the weighted sampler.

---

## 11. Cross-cutting research themes

These are not single-stage tasks. They cut across the whole project.

## 11.1 Blockization research agenda

This is likely the single most important unresolved issue.

Questions:

- What block boundary definition best aligns with PRM stability?
- Should boundaries depend on task type?
- Can a learned boundary detector outperform hand-crafted delimiters?
- How much does block length variance affect ESS and resampling?
- How are code blocks, math equations, and prose reasoning handled consistently?

Experiments:

- compare several blockization schemes on the same model and PRM,
- measure reward increment variance and final correctness,
- quantify boundary jitter sensitivity.

## 11.2 Reward design and calibration

Questions:

- Should the reward estimate future correctness, local correctness, or expected verifier score?
- What positive floor should be imposed on rewards?
- How does calibration drift across datasets or task families?
- Should reward increments be directly regularized during PRM training?

Experiments:

- calibration curves by block depth,
- ablations on reward clipping and flooring,
- different supervision targets.

## 11.3 Chain overlap and temperature-ladder design

Questions:

- how much overlap is needed between neighboring chains,
- how should the ladder be adapted when overlap is too small,
- whether fixed ladders transfer across tasks,
- whether adaptive ladders are worth the complexity.

Experiments:

- estimate bank overlap metrics,
- vary number of chains and spacing,
- track transfer usefulness from chain \(k\) to \(k+1\).

## 11.4 Compute and memory engineering

Questions:

- can KV caches be shared via a prefix DAG,
- when should caches be evicted,
- how expensive is PRM inference relative to sampling,
- can bank storage be compressed,
- what batching strategies are needed for practical throughput.

Experiments:

- benchmark memory growth versus particle count,
- benchmark PRM batch size versus latency,
- benchmark prefix deduplication strategies.

## 11.5 Output selection and deployment semantics

Questions:

- what trajectory should be returned to users or evaluators,
- how much reranking after cold-chain inference is acceptable,
- should one return a posterior sample or a highest-score candidate,
- how does this choice affect reproducibility and benchmark comparability.

This must be explicitly fixed in evaluation.

---

## 12. Major risks and mitigation plan

## 12.1 Risk: PRM dominates performance

If the PRM is noisy or poorly calibrated, the sampler degenerates.

Mitigation:

- dedicate an explicit stage to PRM calibration,
- operate in log-space,
- clip unstable increments,
- monitor reward-increment distributions continuously.

## 12.2 Risk: blockization is unstable

Poor block boundaries create noisy rewards and brittle depth alignment.

Mitigation:

- treat blockization as its own module and experimental axis,
- log every boundary decision,
- compare at least two simple policies before scaling up.

## 12.3 Risk: compute explosion

Many particles, many chains, and repeated PRM calls may be too expensive.

Mitigation:

- establish single-chain baselines first,
- batch PRM scoring,
- cap depth,
- instrument cost from the beginning,
- defer large benchmarks until costs are understood.

## 12.4 Risk: hot-bank communication collapses diversity

Aggressive injection may cause cold chains to become copies of hotter chains.

Mitigation:

- sweep \(\lambda\),
- monitor duplicate rates,
- test deduplication,
- compare communication frequencies.

## 12.5 Risk: scope creep

The document mentions optional twist learning, editing networks, and broad benchmarks.

Mitigation:

- lock the first milestone to online blockwise tempered SMC only,
- require explicit stage completion before expanding scope.

---

## 13. Recommended milestone sequence

### Milestone A
Single-chain blockwise SMC works end-to-end on a small math subset.

### Milestone B
PRM calibration is stable enough for sequential weighting.

### Milestone C
Multiple temperatures run independently and show distinct exploration regimes.

### Milestone D
Hot-bank communication is correct and improves over no-communication baselines.

### Milestone E
Evaluation harness supports fair compute-matched comparisons.

### Milestone F
Optional amortized proposal learning is attempted using the online system as a teacher.

---

## 14. Immediate next actions

The following actions should be executed first in the repository.

1. Write the core interfaces for prefixes, particles, chains, banks, and PRM scoring.
2. Choose and document a **v1 blockization policy**.
3. Implement single-chain blockwise SMC.
4. Add metrics and trace logging from day one.
5. Build a tiny deterministic benchmark slice for debugging.
6. Measure reward-increment statistics before adding any multi-chain complexity.
7. Add temperature support only after Stage 1 is stable.
8. Add hot-bank communication only after Stage 3 is stable.

---

## 15. Definition of done for the first serious release

A first serious release of the repository should satisfy all of the following:

- One-chain blockwise SMC is correct and tested.
- PRM integration is calibrated enough to avoid immediate degeneracy.
- Multi-chain tempering is implemented with configurable ladders.
- Hot-bank communication uses exact branch-aware importance weighting.
- Metrics, lineage traces, and debugging artifacts are preserved.
- Evaluation compares against at least best-of-N and standard SMC under equal compute.
- Optional twist learning remains clearly marked as experimental unless it has its own validation.

---

## 16. Summary

The right build order is:

1. define the state space,
2. implement plain blockwise SMC,
3. stabilize the PRM,
4. add temperatures,
5. add hot-bank injection,
6. evaluate rigorously,
7. only then amortize with learned proposals.

The main unresolved scientific questions are:

- how to define blocks,
- how to train and calibrate the PRM,
- how to space temperatures,
- how aggressively to inject hot prefixes,
- and how to manage compute and memory at scale.

Those questions should be treated as explicit research tasks, not hidden assumptions.
