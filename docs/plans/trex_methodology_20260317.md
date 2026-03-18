
# T-REX Methodology: Blockwise Tempered Sequential Monte Carlo for LLM Reasoning

## Status and scope

This document translates the project note into implementation-oriented methodology documentation. It is written as production-facing technical documentation for engineers and researchers implementing the system.

The central online method is **blockwise interacting tempered Sequential Monte Carlo (SMC)** over variable-length reasoning trajectories. The optional amortized extension is **twisted SMC / learned proposal learning**. The proposal is inspired by parallel tempering, but the actual online algorithm is **not** classical swap-based parallel tempering. It is an interacting tempered SMC scheme that communicates across temperatures through **hot-bank resampling** rather than Metropolis–Hastings swaps.

---

## 1. Problem formulation

### 1.1 Reasoning trajectories as variable-length block sequences

Given an input prompt \(x_0\), we represent a generated solution as a variable-length sequence of reasoning blocks

\[
b_{1:M} = (b_1, \dots, b_M),
\]

where each block \(b_m\) is a variable-length contiguous span of tokens ending at a semantic boundary, such as:

- a completed reasoning step,
- an equation block,
- a derivation fragment,
- a structured explanation segment,
- or an explicit answer block.

The stopping time \(M\) is random. A completed trajectory is a sequence that terminates in a designated end-of-solution event.

Let \(\mathcal{B}_M\) denote the set of valid completed block sequences of length \(M\), and define the full trajectory space as

\[
\mathcal{B} = \bigcup_{M \ge 0} \mathcal{B}_M.
\]

The base language model induces a prior over completed trajectories:

\[
P_\theta(b_{1:M} \mid x_0).
\]

For brevity, we suppress the conditioning on \(x_0\) when it is clear from context and write \(P_\theta(b_{1:M})\).

### 1.2 Hard constraints versus soft process rewards

A verifier-based formulation would often impose a hard terminal constraint

\[
\phi(b_{1:M}) \in \{0,1\},
\]

yielding a posterior proportional to

\[
\pi(b_{1:M}) \propto P_\theta(b_{1:M}) \phi(b_{1:M}).
\]

That formulation is brittle in text spaces because nearly-correct partial trajectories receive zero mass before completion. The proposal instead replaces hard rejection with a **strictly positive block-prefix reward model**

\[
R(b_{1:m}) \in (0,\infty),
\]

evaluated only at completed block boundaries.

The target posterior becomes

\[
\pi(b_{1:M}) = \frac{1}{Z_\pi} P_\theta(b_{1:M}) R(b_{1:M}),
\]

with normalizing constant

\[
Z_\pi = \sum_{M \ge 0} \sum_{b_{1:M} \in \mathcal{B}_M} P_\theta(b_{1:M}) R(b_{1:M}).
\]

Interpretation:

- \(P_\theta\) supplies the base model prior over reasoning trajectories.
- \(R\) supplies a soft potential expressing prefix quality or completability.
- The posterior preserves multiple modes rather than collapsing onto a single highest-likelihood path.

---

## 2. Absorbing-state embedding for variable-length reasoning

To place all particles on a shared sequential time axis, terminated trajectories are embedded into an absorbing process.

Introduce a terminal block symbol \(\dagger\). Once a trajectory terminates,

\[
b_{m+1} = \dagger \implies b_{m+2} = \dagger, b_{m+3} = \dagger, \dots
\]

This allows every particle to evolve on a common block index \(m = 0,1,2,\dots\), while terminated particles remain fixed.

This embedding is operationally important because it lets us:

- define a single SMC recursion over all particles,
- compute ESS and resampling decisions at common depth indices,
- align chain-to-chain communication at the same block depth,
- and avoid separate bookkeeping for variable trajectory lengths.

---

## 3. Base blockwise Sequential Monte Carlo

## 3.1 Intermediate prefix targets

Define the unnormalized prefix target at block depth \(m\) by

\[
\widetilde{\pi}_m(b_{1:m}) = P_\theta(b_{1:m}) R(b_{1:m}), \qquad m \ge 1,
\]

with conventions

\[
R(b_{1:0}) \equiv 1, \qquad \widetilde{\pi}_0(\varnothing) \equiv 1.
\]

The goal is to approximate the evolving sequence of prefix marginals induced by these targets.

## 3.2 Propagation

SMC maintains \(N\) particles. At block boundary \(m\), particle \(i\) is propagated by sampling a new block from the base model:

\[
b_m^{(i)} \sim P_\theta(\cdot \mid b_{1:m-1}^{(i)}).
\]

This means the proposal is the base model itself at the block level.

## 3.3 Incremental importance weighting

The incremental importance weight is

\[
w_m^{(i)}
=
\frac{\widetilde{\pi}_m(b_{1:m}^{(i)})}
{\widetilde{\pi}_{m-1}(b_{1:m-1}^{(i)}) P_\theta(b_m^{(i)} \mid b_{1:m-1}^{(i)})}.
\]

Expanding the target yields

\[
w_m^{(i)}
=
\frac{P_\theta(b_{1:m}^{(i)})R(b_{1:m}^{(i)})}
{P_\theta(b_{1:m-1}^{(i)})R(b_{1:m-1}^{(i)})P_\theta(b_m^{(i)} \mid b_{1:m-1}^{(i)})}.
\]

Since

\[
P_\theta(b_{1:m}^{(i)}) = P_\theta(b_{1:m-1}^{(i)}) P_\theta(b_m^{(i)} \mid b_{1:m-1}^{(i)}),
\]

the base-model factors cancel exactly:

\[
w_m^{(i)} = \frac{R(b_{1:m}^{(i)})}{R(b_{1:m-1}^{(i)})}.
\]

This is one of the most important identities in the method. It implies:

- the target is shaped by the base model prior,
- but the sequential correction under base-model propagation depends only on **reward increments**,
- so the quality and calibration of \(R\) are central to the algorithm’s behavior.

## 3.4 Weight accumulation and resampling

Let \(W_m^{(i)}\) denote the accumulated particle weight after processing depth \(m\). In the simplest form:

\[
W_m^{(i)} \propto W_{m-1}^{(i)} \, w_m^{(i)}.
\]

After normalization,

\[
\bar{W}_m^{(i)} = \frac{W_m^{(i)}}{\sum_{j=1}^N W_m^{(j)}}.
\]

Define effective sample size:

\[
\mathrm{ESS}_m = \frac{1}{\sum_{i=1}^N (\bar{W}_m^{(i)})^2}.
\]

When ESS falls below a threshold, resample particles according to normalized weights.

Production requirements:

- compute in log-space where possible,
- use stable normalization,
- make the ESS threshold configurable,
- record both pre-resample and post-resample particle ancestry,
- preserve enough lineage information for debugging and training downstream models.

---

## 4. Why the state space is blockwise instead of tokenwise

The method is explicitly defined on completed reasoning blocks rather than arbitrary token prefixes.

This choice has several consequences.

### 4.1 Reward stability

A reward assigned to a completed semantic step is more stable and semantically meaningful than a reward assigned to a partially-written token span.

### 4.2 Lower interaction frequency

PRM evaluation, ESS computation, resampling, and cross-temperature communication happen only at block boundaries, not after every token. This dramatically reduces overhead.

### 4.3 Better alignment with reasoning structure

Particle survival reflects the quality of coherent partial derivations rather than token-level formatting noise.

### 4.4 Blockization is determined by SFT data conventions

A block is a reasoning step. The block boundary — the delimiter that signals the end of a reasoning step — is defined by the conventions of the SFT training data used to fine-tune the base model. After SFT, the model learns to emit these delimiters naturally, so the blockizer simply detects the delimiter pattern that the SFT data established.

This means blockization is **not** an independent open design variable. It is resolved by the choice of SFT dataset and the reasoning step structure therein. The blockizer implementation must match the SFT data's delimiter conventions exactly.

---

## 5. Tempered posterior family

A single SMC system may become trapped in locally plausible but globally wrong reasoning modes. To improve exploration, the proposal defines a temperature ladder

\[
0 = \beta_1 < \beta_2 < \cdots < \beta_K = 1.
\]

For each chain \(k\), define the unnormalized target at depth \(m\):

\[
\widetilde{\pi}_m^k(b_{1:m}) = P_\theta(b_{1:m}) R(b_{1:m})^{\beta_k}.
\]

The corresponding normalized target is

\[
\pi_m^k(b_{1:m}) = \frac{\widetilde{\pi}_m^k(b_{1:m})}{Z_m^k}.
\]

Interpretation of the ladder:

- **Hot chain** (\(\beta_1 = 0\)): target reduces to the base model prior \(P_\theta\).
- **Cold chain** (\(\beta_K = 1\)): target equals the PRM-shaped posterior of actual interest.
- Intermediate chains interpolate between prior-driven exploration and reward-focused exploitation.

Important clarification:

This is **reward tempering**, not necessarily decoding-temperature scaling. The hottest chain removes the reward potential; it does not automatically imply higher token sampling temperature unless the implementation separately chooses to vary decoding temperature.

---

## 6. Inter-chain communication via hot-bank resampling

## 6.1 Why not classical swap-based parallel tempering

Classical parallel tempering communicates through swap proposals between neighboring temperatures, typically accepted or rejected by Metropolis–Hastings criteria.

In discrete, high-dimensional text spaces:

- neighboring typical sets may overlap weakly,
- swap acceptance can collapse,
- and accepted communication may become too rare to justify the machinery.

The proposal replaces swap moves with **direct resampling from a hotter weighted bank**.

This is a key conceptual point: the method is inspired by parallel tempering, but the actual communication mechanism is an interacting SMC construction.

## 6.2 Empirical particle banks

At chain \(k\) and block depth \(m\), let the weighted particle set be

\[
\{b_{1:m}^{k,(i)}, W_m^{k,(i)}\}_{i=1}^N.
\]

Normalize weights:

\[
\bar{W}_m^{k,(i)} = \frac{W_m^{k,(i)}}{\sum_{j=1}^N W_m^{k,(j)}}.
\]

Define the empirical measure

\[
\widehat{\pi}_m^k = \sum_{i=1}^N \bar{W}_m^{k,(i)} \, \delta_{b_{1:m}^{k,(i)}}.
\]

Operationally, \(\widehat{\pi}_m^k\) is a **finite bank** of completed prefixes. It should not be interpreted as a general-purpose density lookup over arbitrary text prefixes.

## 6.3 Two-branch proposal for the colder chain

For adjacent chains \(k\) and \(k+1\), with \(\beta_k < \beta_{k+1}\), the colder chain at depth \(m\) samples a branch indicator

\[
C_m \in \{\mathrm{loc}, \mathrm{hot}\},
\quad
\Pr(C_m=\mathrm{loc}) = 1-\lambda,
\quad
\Pr(C_m=\mathrm{hot}) = \lambda,
\]

where \(\lambda \in [0,1]\) is a communication hyperparameter.

### Local branch

With probability \(1-\lambda\), sample a parent from the cold bank at depth \(m-1\) and extend it by one new block:

\[
q_{m}^{\mathrm{loc},k+1}(b_{1:m})
=
\widehat{\pi}_{m-1}^{k+1}(b_{1:m-1}) P_\theta(b_m \mid b_{1:m-1}).
\]

### Hot branch

With probability \(\lambda\), copy a completed prefix from the hotter bank at the same depth:

\[
q_{m}^{\mathrm{hot},k\to k+1}(b_{1:m}) = \widehat{\pi}_m^k(b_{1:m}).
\]

Operationally, hot-branch sampling is

\[
J \sim \mathrm{Cat}(\bar{W}_m^{k,(1)}, \dots, \bar{W}_m^{k,(N)}),
\qquad
b_{1:m} = b_{1:m}^{k,(J)}.
\]

This mechanism transports promising prefixes downward in temperature without requiring explicit swap acceptance.

---

## 7. Exact weighting views for the two-branch proposal

The proposal admits two mathematically valid importance-sampling interpretations.

## 7.1 View A: mixture importance sampling on prefixes

Treat the marginal proposal on \(b_{1:m}\) as

\[
q_m^{k+1}(b_{1:m})
=
(1-\lambda) q_m^{\mathrm{loc},k+1}(b_{1:m})
+
\lambda q_m^{\mathrm{hot},k\to k+1}(b_{1:m}).
\]

Then the exact importance weight targeting \(\widetilde{\pi}_m^{k+1}\) is

\[
w_m^{k+1}(b_{1:m})
=
\frac{\widetilde{\pi}_m^{k+1}(b_{1:m})}
{(1-\lambda) q_m^{\mathrm{loc},k+1}(b_{1:m}) + \lambda q_m^{\mathrm{hot},k\to k+1}(b_{1:m})}.
\]

This is standard mixture importance sampling.

## 7.2 View B: augmented-state importance sampling

Treat the sampled branch indicator as part of the state. The augmented state is

\[
(C_m, b_{1:m}) \in \{\mathrm{loc}, \mathrm{hot}\} \times \mathcal{B}_m.
\]

Define the augmented proposal

\[
r_m^{k+1}(c,b_{1:m}) = \rho(c) q_m^c(b_{1:m}),
\]

with

\[
\rho(\mathrm{loc}) = 1-\lambda,
\qquad
\rho(\mathrm{hot}) = \lambda.
\]

Define the augmented unnormalized target

\[
\overline{\pi}_m^{k+1}(c,b_{1:m}) = \rho(c)\widetilde{\pi}_m^{k+1}(b_{1:m}).
\]

Marginalizing over \(c\) recovers the original target on prefixes.

Then the exact importance weight is

\[
w_m^{k+1}(c,b_{1:m})
=
\frac{\overline{\pi}_m^{k+1}(c,b_{1:m})}{r_m^{k+1}(c,b_{1:m})}
=
\frac{\widetilde{\pi}_m^{k+1}(b_{1:m})}{q_m^c(b_{1:m})}.
\]

This explains the statement that branch probabilities “cancel”: they cancel because they appear in both the augmented proposal and the augmented target.

## 7.3 Why the implementation should use the augmented view

The augmented view is preferable in production because:

- it matches the actual sampling procedure,
- it avoids evaluating both branches for every sampled prefix,
- it gives branch-specific corrections directly,
- and it avoids brittle lookup requirements.

---

## 8. Branch-specific importance corrections

Under the augmented-state formulation, the exact empirical corrections are:

### Local branch

\[
w_m^{k+1}(\mathrm{loc}, b_{1:m})
=
\frac{\widetilde{\pi}_m^{k+1}(b_{1:m})}
{\widehat{\pi}_{m-1}^{k+1}(b_{1:m-1}) P_\theta(b_m \mid b_{1:m-1})}.
\]

### Hot branch

\[
w_m^{k+1}(\mathrm{hot}, b_{1:m})
=
\frac{\widetilde{\pi}_m^{k+1}(b_{1:m})}
{\widehat{\pi}_{m}^{k}(b_{1:m})}.
\]

These are exact with respect to the finite empirical proposal kernels that the algorithm actually uses.

---

## 9. Idealized simplifications

To interpret the empirical formulas, replace empirical banks by exact tempered prefix targets:

\[
\widehat{\pi}_{m-1}^{k+1} \rightsquigarrow \pi_{m-1}^{k+1},
\qquad
\widehat{\pi}_{m}^{k} \rightsquigarrow \pi_{m}^{k}.
\]

## 9.1 Local branch simplification

\[
w_m^{k+1}(\mathrm{loc}, b_{1:m})
=
\frac{P_\theta(b_{1:m}) R(b_{1:m})^{\beta_{k+1}}}
{\pi_{m-1}^{k+1}(b_{1:m-1}) P_\theta(b_m \mid b_{1:m-1})}.
\]

Substituting

\[
\pi_{m-1}^{k+1}(b_{1:m-1})
=
\frac{P_\theta(b_{1:m-1})R(b_{1:m-1})^{\beta_{k+1}}}{Z_{m-1}^{k+1}},
\]

gives

\[
w_m^{k+1}(\mathrm{loc}, b_{1:m})
=
Z_{m-1}^{k+1}
\left(
\frac{R(b_{1:m})}{R(b_{1:m-1})}
\right)^{\beta_{k+1}}.
\]

Hence, up to a depth-dependent constant,

\[
w_m^{k+1}(\mathrm{loc}, b_{1:m})
\propto
\left(
\frac{R(b_{1:m})}{R(b_{1:m-1})}
\right)^{\beta_{k+1}}.
\]

## 9.2 Hot branch simplification

\[
w_m^{k+1}(\mathrm{hot}, b_{1:m})
=
\frac{P_\theta(b_{1:m})R(b_{1:m})^{\beta_{k+1}}}
{\pi_m^k(b_{1:m})}.
\]

Substituting

\[
\pi_m^k(b_{1:m})
=
\frac{P_\theta(b_{1:m})R(b_{1:m})^{\beta_k}}{Z_m^k},
\]

gives

\[
w_m^{k+1}(\mathrm{hot}, b_{1:m})
=
Z_m^k R(b_{1:m})^{\beta_{k+1}-\beta_k}.
\]

Hence, up to a depth-dependent constant,

\[
w_m^{k+1}(\mathrm{hot}, b_{1:m})
\propto
R(b_{1:m})^{\beta_{k+1}-\beta_k}.
\]

Interpretation:

- local-branch correction depends on the **incremental reward ratio** raised to the colder inverse temperature;
- hot-branch correction depends on the **absolute prefix reward** raised to the temperature gap.

---

## 10. Exactness, asymptotics, and what is actually guaranteed

The method has two levels of correctness.

### 10.1 Exact relative to the actual empirical proposal

Conditioned on the empirical banks used to construct proposals, the branch-specific corrections above are exact importance weights for the proposal kernels that are actually sampled.

### 10.2 Not exactly unbiased relative to the ideal population flow at finite \(N\)

Because the proposal itself depends on empirical approximations \(\widehat{\pi}\), the finite-particle system is an adaptive/interacting SMC algorithm. Therefore:

- the method is not exactly identical to a population algorithm at finite \(N\),
- and exact unbiasedness relative to the idealized target flow is not available in the naïve sense.

### 10.3 Asymptotic consistency

Under standard adaptive/interacting SMC regularity conditions, including:

- consistent resampling,
- positive bounded potentials,
- sufficient support coverage,
- and stable proposal construction,

the empirical measures converge to their population counterparts as \(N \to \infty\).

Thus, the practical algorithm is asymptotically consistent.

### 10.4 Important claim discipline

Documentation and papers should avoid claiming that this is exact classical parallel tempering. It is more accurate to call it:

- blockwise interacting tempered SMC,
- hot-bank resampling tempered SMC,
- or a replica-exchange-inspired interacting SMC algorithm.

---

## 11. Why hot-bank resampling avoids brittle prefix lookup

A misleading way to read the method would be to think that the empirical measure \(\widehat{\pi}_m^k\) must support arbitrary exact-match prefix lookup in the huge discrete text space. That is not required.

Under the actual implementation:

- the local branch only needs the sampled cold parent and the sampled block extension,
- the hot branch only needs the weight of a prefix that is explicitly present in the hotter bank.

So the system never requires evaluating the empirical mass of an arbitrary queried prefix. This is a practical advantage and removes a major source of brittleness.

---

## 12. Process reward model requirements

The PRM is not a secondary reranker in this design. It is the sequential potential that shapes every incremental weight. Because the base-model factors cancel, the PRM is one of the highest-leverage components in the entire system.

### 12.1 Required properties

A useful PRM for blockwise SMC should satisfy at least four properties.

#### Prefix sensitivity
It should distinguish prefixes that remain plausibly completable from prefixes that already contain fatal errors.

#### Cross-depth comparability
Scores at different block depths must be sufficiently calibrated because the algorithm uses ratios \(R(b_{1:m})/R(b_{1:m-1})\).

#### Boundary stability
Completed block boundaries should smooth away small internal token perturbations; reward should not swing wildly due to formatting noise.

#### Non-vanishing support
Recoverable but imperfect prefixes should not receive numerically zero reward, or the method degenerates back into hard rejection.

### 12.2 Training targets

Natural supervision targets include:

- binary eventual correctness,
- soft success probability,
- pairwise prefix preference,
- reward-to-go estimates from sampled suffix completions.

The most natural semantic target is **probability of eventual successful completion from the current completed prefix**, not merely local stylistic correctness.

### 12.3 Calibration for sequential weighting

A ranking-only PRM may be insufficient. Sequential weighting requires stable increments.

Recommended implementation pattern:

1. Train in log-space:
   \[
   \ell(b_{1:m}) = \log R(b_{1:m}).
   \]

2. Compute increments:
   \[
   \Delta \ell_m = \ell(b_{1:m}) - \ell(b_{1:m-1}).
   \]

3. Normalize, clip, or regularize \(\Delta \ell_m\) to control volatility.

4. Exponentiate only after stabilization if actual reward values are needed.

### 12.4 Training-time and test-time blockization must match

Because reward is defined on completed block boundaries, PRM data must use the same block structure — i.e., the reasoning step delimiter from the SFT training data — as deployed during inference. The PRM must be trained on outputs from the SFT'd model, not the raw base model. Segmentation mismatch is a direct calibration failure mode.

---

## 13. Optional amortized extension: twisted SMC

The main methodology above is a pure inference-time method. The appendix proposes an optional amortized extension.

## 13.1 Twisted proposal

For inverse temperature \(\beta \in (0,1]\), define

\[
\widetilde{\pi}_m^\beta(b_{1:m}) = P_\theta(b_{1:m}) R(b_{1:m})^\beta.
\]

Introduce positive twist potentials \(\psi_{\gamma,m}(b_{1:m}) > 0\). Then define a twisted block proposal

\[
q_{\gamma,m}(b_m \mid b_{1:m-1})
=
P_\theta(b_m \mid b_{1:m-1})
\frac{\psi_{\gamma,m}(b_{1:m})}{Z_{\gamma,m}(b_{1:m-1})},
\]

where

\[
Z_{\gamma,m}(b_{1:m-1})
=
\sum_{b_m'} P_\theta(b_m' \mid b_{1:m-1}) \psi_{\gamma,m}(b_{1:m-1}, b_m').
\]

Twisting changes the proposal, not the target.

## 13.2 Correct incremental weight under twisting

Using the twisted proposal gives

\[
w_m^\beta(b_{1:m})
=
\frac{\widetilde{\pi}_m^\beta(b_{1:m})}
{\widetilde{\pi}_{m-1}^\beta(b_{1:m-1}) q_{\gamma,m}(b_m \mid b_{1:m-1})}
=
\left(
\frac{R(b_{1:m})}{R(b_{1:m-1})}
\right)^\beta
\frac{Z_{\gamma,m}(b_{1:m-1})}{\psi_{\gamma,m}(b_{1:m})}.
\]

Thus the sampler remains properly weighted so long as the Radon–Nikodym correction is applied.

## 13.3 Purpose of the twist

The ideal fully-adapted proposal depends on an intractable reward-to-go term. The twist serves as a tractable surrogate for that reward-to-go. In the ideal case, the twist reduces the variance of incremental weights and makes resampling less necessary.

---

## 14. Curriculum-tempered twist learning

The proposal also suggests using the same temperature family as a curriculum for offline proposal learning.

### 14.1 Temperature-conditioned proposal

For \(\beta \in [0,1]\),

\[
\widetilde{\pi}_\beta(b_{1:M}) \propto P_\theta(b_{1:M}) R(b_{1:M})^\beta.
\]

Train a \(\beta\)-conditioned proposal

\[
q_\gamma(b_{1:M} \mid \beta)
=
\prod_{m=1}^M q_{\gamma,m}(b_m \mid b_{1:m-1}, \beta).
\]

### 14.2 Forward-KL objective

A natural amortization objective is forward KL from the cold posterior to the learned proposal:

\[
D_{\mathrm{KL}}(\pi \,\|\, q_\gamma)
=
\mathbb{E}_{\pi}\left[\log \pi(B_{1:M}) - \log q_\gamma(B_{1:M})\right].
\]

Dropping terms constant in \(\gamma\), this becomes

\[
\arg\min_\gamma D_{\mathrm{KL}}(\pi \,\|\, q_\gamma)
\equiv
\arg\max_\gamma \mathbb{E}_{\pi}\left[\sum_{m=1}^M \log q_{\gamma,m}(B_m \mid B_{1:m-1})\right].
\]

With approximate posterior samples \(\{b_{1:M}^{(i)}, w^{(i)}\}_{i=1}^L\), use self-normalized weighted maximum likelihood:

\[
\max_\gamma \sum_{i=1}^L \bar{w}^{(i)}
\left[
\sum_{m=1}^M \log q_{\gamma,m}(b_m^{(i)} \mid b_{1:m-1}^{(i)})
\right],
\qquad
\bar{w}^{(i)} = \frac{w^{(i)}}{\sum_j w^{(j)}}.
\]

### 14.3 Why forward KL

Forward KL is mass-covering. It encourages the learned proposal to allocate probability across the posterior support rather than collapsing to a single dominant mode.

### 14.4 Curriculum over temperatures

Use a schedule \(\rho_t(\beta)\) over training stages \(t\), starting with hotter targets and gradually concentrating near \(\beta = 1\). This acts like continuation or homotopy training: solve easy smoothed problems first, then move toward the sharp target of actual interest.

---

## 15. System components implied by the methodology

An implementation faithful to the proposal needs at least the following components.

## 15.1 Blockizer
Responsible for:

- detecting the reasoning step delimiter established by the SFT training data,
- deciding when a block terminates (i.e., when the delimiter pattern is emitted),
- serializing blocks,
- handling EOS / final answer blocks,
- ensuring deterministic reconstruction from token streams.

The blockizer does not define block boundaries independently. It detects the delimiter pattern that the SFT'd model has learned to emit at reasoning step boundaries.

## 15.2 Base block sampler
Responsible for:

- generating one block conditioned on a prefix,
- managing stopping rules for block generation,
- exposing token logprobs if needed,
- maintaining KV-cache state for many particles.

### 15.2.1 Base model SFT: foundational requirement

The entire SMC framework is shaped by the base model prior \(P_\theta\). If \(P_\theta\) assigns negligible mass to correct proof-style reasoning, the posterior \(\pi\) requires extreme importance-weight corrections that are impractical at finite \(N\). SFT on high-quality proof data is **required** to shift \(P_\theta\) toward the target reasoning style before any SMC inference is meaningful.

The SFT dataset for this purpose is **FineProofs-SFT** ([lm-provers/FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT)):

- 7,777 samples (4,300 unique problems) from olympiad competitions and Art of Problem Solving.
- Includes chain-of-thought reasoning traces and formal proofs.
- Covers algebra, combinatorics, number theory, geometry, and inequalities.
- Provides expert quality grades (0–7 scale) and reward scores suitable for curriculum-based SFT.
- Decontaminated against standard proof benchmarks (IMOProofBench, ProofBench).

The SFT'd model replaces the raw base model as \(P_\theta\) in all subsequent SMC stages.

## 15.3 Prefix reward model
Responsible for:

- scoring completed prefixes,
- outputting stable positive potentials or log-potentials,
- supporting batch inference,
- supporting calibration and monitoring.

## 15.4 SMC engine
Responsible for:

- per-particle propagation,
- incremental weight updates,
- ESS computation,
- resampling,
- ancestry tracking,
- termination handling.

## 15.5 Tempering manager
Responsible for:

- temperature ladder definition,
- inter-chain scheduling,
- synchronization at block boundaries,
- chain-local metrics and logging.

## 15.6 Hot-bank manager
Responsible for:

- storing weighted prefix banks,
- exposing them to colder chains,
- branch sampling,
- deduplication and memory control,
- implementing bank refresh semantics.

## 15.7 Optional twist/proposal learner
Responsible for:

- proposal parameterization,
- teacher data extraction from particle trajectories,
- temperature-conditioned training,
- offline validation,
- optional online use during inference.

---

## 16. What the method does not currently specify

The methodology is mathematically coherent, but several implementation details are intentionally left open.

1. ~~Exact blockization policy.~~ Resolved: blockization is determined by the SFT data's reasoning step delimiter conventions (see Section 4.4).
2. Exact block-sampling mechanism from the base model.
3. Output selection rule at the end of inference.
4. Hyperparameter defaults for \(K\), \(N\), \(\lambda\), ESS threshold, and max depth.
5. Memory policy for many cached prefixes and per-particle states.
6. Synchronization policy when chains or particles terminate at different times.
7. Distributed execution model.
8. Precise PRM architecture and training objective.
9. Whether token decoding temperature differs across chains.
10. Whether optional components such as twist learning or editing networks are in scope for the first implementation.

These omissions are not bugs in the mathematical proposal, but they are major design tasks for the repository implementation.

---

## 17. Minimal algorithm sketch

```text
Input:
  prompt x0
  base model P_theta
  prefix reward model R
  temperatures 0 = beta_1 < ... < beta_K = 1
  particles per chain N
  hot-branch probability lambda
  ESS threshold tau

Initialize:
  For each chain k:
    create N empty-prefix particles with weight 1

For depth m = 1, 2, ... until all particles absorbed or max depth reached:
  For chain k = 1:
    propagate locally from P_theta at depth m
    score completed prefixes with R
    update weights using reward-ratio rule raised to beta_1 (= 0 in hottest chain target)
    resample if ESS < tau
    store weighted bank at depth m

  For each colder chain k+1:
    for each particle:
      sample branch C in {loc, hot}
      if loc:
        sample parent from current cold bank at depth m-1
        sample next block from P_theta
        weight with exact local-branch correction
      if hot:
        sample completed prefix from hotter bank at depth m
        weight with exact hot-branch correction
    normalize weights
    resample if ESS < tau
    store weighted bank at depth m

Return:
  completed trajectories from cold chain, plus metadata for selection/reranking.
```

This sketch should not be treated as the full implementation specification; it omits many engineering details covered elsewhere in this repository.

---

## 18. Recommended reading order for implementers

1. Read the posterior definition and absorbing-state embedding.
2. SFT the base model on proof data (FineProofs-SFT). Identify the reasoning step delimiter from the SFT data — this defines block boundaries.
3. Build the blockizer around the SFT-derived delimiter.
4. Implement plain blockwise SMC.
5. Add temperature ladder support without communication.
6. Add hot-bank communication and branch-specific weighting.
7. Only after the core sampler is stable, consider amortized twist learning.

---

## 19. Summary

The core method can be summarized as follows:

- Model reasoning as a variable-length sequence of semantic blocks.
- Define a soft posterior over reasoning trajectories using the base model prior and a strictly positive prefix reward model.
- Approximate that posterior with blockwise SMC, where incremental weights are driven by reward ratios.
- Introduce a temperature ladder over reward strength to broaden exploration.
- Replace brittle swap-based cross-temperature communication with direct hot-bank resampling from hotter empirical particle banks.
- Optionally use the resulting particle system as a teacher for learning reusable twisted proposals.

The method is strongest when the repository treats the following as first-class systems:

- base model SFT (which determines both prior quality and block structure),
- blockization (derived from SFT data delimiter conventions),
- PRM calibration (trained on SFT'd model outputs),
- exact branch-aware importance weighting,
- bank management,
- and stable particle systems engineering.
