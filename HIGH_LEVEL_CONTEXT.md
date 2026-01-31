# System Context & Mathematical Specification for T-REX: Twisted Replica Exchange for Bootstrapping Reasoning

## 1. Abstract Framework

T-REX is a probabilistic inference framework designed to solve the "Narrow Passage" problem in constrained language generation (e.g., math reasoning, code generation). It decouples the reasoning process into three distinct, interacting mechanisms:

- **Relaxation (Exploration):** A thermodynamic search engine that maintains parallel chains at high temperatures. These chains relax the strict constraints of the target distribution, allowing the model to "dream" and discover diverse semantic structures ("Proto-solutions") that would be rejected by standard decoding.

- **Transport (Discrete Editor):** A non-reversible bridge that projects high-temperature samples onto the constrained manifold. By performing local, valid-by-construction repairs (via Block-Gibbs sampling), it enables the efficient transfer of "hot" ideas into "cold" valid solutions, bypassing the high-energy barriers of the problem landscape.

- **Guidance (Twisted Distillation):** A feedback loop that distills the successful trajectories found by the transport mechanism into a global value function (Twist). This biases future proposals, progressively converting the expensive search process into efficient intuition (System 1).

## 2. Mathematical Formulation

### 2.1. The Target Distribution

We formulate language generation as sampling from an unnormalized target distribution $\sigma(x)$ over token sequences $x = (x_1, \dots, x_T)$:

$$\sigma(x) \propto p_0(x) \cdot \phi(x)$$

- $p_0(x) = \prod_{t=1}^T p_\theta(x_t | x_{<t})$: The pre-trained Base LLM (the prior).

- $\phi(x) \in [0, 1]$: The non-local potential function (constraint/verifier). For binary constraints, $\phi(x) = \mathbb{I}(x \text{ is valid})$.

- $\mathcal{Z}$: The intractable normalization constant (partition function).

### 2.2. Parallel Tempering (The Landscape)

We maintain $K$ parallel chains targeting a sequence of annealed distributions $\pi_k(x)$ defined by inverse temperatures $0 = \beta_0 < \beta_1 < \dots < \beta_K = 1$:

$$\pi_k(x) \propto p_0(x) \cdot \phi(x)^{\beta_k}$$

- $\beta \approx 0$ (Hot): $\pi_0(x) \approx p_0(x)$. Pure exploration driven by the base model's priors. High diversity, low validity.

- $\beta = 1$ (Cold): $\pi_K(x) \propto p_0(x)\phi(x)$. Strict sampling from the posterior. High validity, low diversity.

### 2.3. Non-Reversible Parallel Tempering (Ballistic Flow)

**The Random Walk Problem:** In standard parallel tempering, swap pairs are chosen at random. This causes particles to perform a random walk through the temperature ladder, requiring $O(K^2)$ steps for a sample to propagate from the prior ($\beta_0$) to the posterior ($\beta_K$).

**The Solution:** Following Syed et al. (2022), we use a **deterministic, alternating schedule** instead of random swap selection:

$$\mathcal{S}_{odd} = \{(1,2), (3,4), (5,6), \dots\}$$
$$\mathcal{S}_{even} = \{(2,3), (4,5), (6,7), \dots\}$$

At odd timesteps, we attempt swaps between pairs in $\mathcal{S}_{odd}$. At even timesteps, we attempt swaps between pairs in $\mathcal{S}_{even}$.

**Ballistic Flow:** By toggling between these sets, we induce a directed, ballistic flow through the temperature ladder. A successful particle propagates from the prior to the posterior in $O(K)$ steps—significantly faster than the diffusive $O(K^2)$ of standard PT.

**Intuition:** Think of it like a bucket brigade vs. random passing. In standard PT, a bucket (sample) gets passed randomly left or right. In non-reversible PT, buckets consistently move in one direction, creating an efficient pipeline.

### 2.4. The Twist Function (Guidance)

To guide the base model toward high-value regions efficiently, we introduce a Twist Function $\psi_\gamma(x_{1:t})$. Theoretically, the optimal twist function $\psi^*$ corresponds to the expected future potential of the sequence:

$$\psi^*(x_{1:t}) = \mathbb{E}_{p_0(x_{t+1:T}|x_{1:t})} [\phi(x_{1:T})]$$

In implementation, we approximate this using a single global Value Head $V_\gamma(h_t)$ (where $h_t$ is the LLM hidden state at step $t$). The Twisted Proposal Distribution $q_t$ is defined as:

$$q_t(x_t | x_{<t}) \propto p_0(x_t | x_{<t}) \cdot \psi_\gamma(x_{1:t})$$

This tilts the next-token probability mass toward tokens that lead to high-value futures.

## 3. The Transport Mechanism: Adaptive Block-Gibbs

Standard Replica Exchange fails in text domains because the overlap between valid and invalid distributions is negligible. T-REX solves this via an Editor that performs local repairs to bridge the gap.

### 3.1. The Editor Kernel

The editor defines a transition kernel $T(x'|x)$ consisting of:

- **Critic ($C_\omega$):** Selects a mask $m \in \{0, 1\}^T$ identifying error tokens.

- **Proposer:** Resamples masked tokens from the base prior $p_0$ conditioned on the unmasked context.

$$x' \sim p_0(x_{mask} | x_{\neg mask})$$

### 3.2. Acceptance Ratio (The "Free Lunch")

When proposing a move from a current state $x$ to a candidate $x'$ (e.g., swapping a repaired hot sample into a cold chain), the Metropolis-Hastings acceptance ratio $\alpha$ is:

$$\alpha = \min \left(1, \frac{\pi_{target}(x')}{\pi_{target}(x)} \frac{q(x|x')}{q(x'|x)} \right)$$

Substituting the target definition $\pi(x) \propto p_0(x)\phi(x)^\beta$ and the proposal probabilities:

**Ratio of Targets:**

$$\frac{\pi(x')}{\pi(x)} = \frac{p_0(x') \phi(x')^\beta}{p_0(x) \phi(x)^\beta}$$

**Ratio of Proposals:**
Since $q(x'|x) = p_0(x'_{mask} | x_{\neg mask})$ and $p_0(x) = p_0(x_{\neg mask})p_0(x_{mask}|x_{\neg mask})$, the ratio simplifies nicely because the unmasked context probability $p_0(x_{\neg mask})$ is identical for both $x$ and $x'$:

$$\frac{q(x|x')}{q(x'|x)} = \frac{p_0(x_{mask} | x_{\neg mask})}{p_0(x'_{mask} | x_{\neg mask})} = \frac{p_0(x)/p_0(x_{\neg mask})}{p_0(x')/p_0(x_{\neg mask})} = \frac{p_0(x)}{p_0(x')}$$

**Crucial Cancellation:**
Multiplying the terms, the expensive likelihoods $p_0(x)$ and $p_0(x')$ cancel out perfectly:

$$\alpha = \min \left(1, \left( \frac{\phi(x')}{\phi(x)} \right)^{\beta_{target}} \right)$$

**Implication:** We achieve mathematically valid probabilistic inference without computing forward log-probabilities of the full sequence. The acceptance depends solely on the ratio of constraint satisfactions (the potential $\phi$).

## 4. T-REX Algorithm Lifecycle

**Inputs:** Base LLM $p_\theta$, Critic $C_\omega$, Value Head $V_\gamma$.

### Phase 1: Twisted SMC (Exploration)

Run $K$ parallel chains. For each step $t$:

- **Lookahead:** Estimate value $V_\gamma(x_{1:t})$.

- **Propose:** Sample $x_{t+1} \sim \text{Softmax}(\log p_0 + \log V_\gamma)$.

- **Resample:** Periodically duplicate high-value particles and prune low-value ones within the same temperature rung.

### Phase 2: Non-Reversible Transport (Communication)

At fixed intervals, attempt to promote a sample $x_{hot}$ from chain $k$ to chain $k+1$:

- **Edit:** $x' \leftarrow \text{Editor}(x_{hot})$ (Mask & Infill).

- **Swap/Promote:** Treat $x'$ as a candidate for chain $k+1$.

- **Accept:** If $\phi(x')^{\beta_{k+1}} > U[0,1] \cdot \phi(x_{current})^{\beta_{k+1}}$, replace $x_{current}$ with $x'$.

### Phase 3: Online Learning (Self-Distillation)

We close the loop by training the auxiliary networks on the data generated by the search.

- **Dataset:** $\mathcal{D} = \{ (x_{1:t}, y_t) \}$ where $y_t$ is the realized final reward $\phi(x_{1:T})$.

- **Train Twist ($V_\gamma$):** Minimize MSE between predicted value and realized reward (Self-Distillation).

$$\mathcal{L}_{Twist} = || V_\gamma(x_{1:t}) - \phi(x_{1:T}) ||^2$$

- **Train Critic ($C_\omega$):** Maximize likelihood of masks that led to successful repairs.

## 5. Implementation Architecture

### 5.1. Model Components

- **Base Model:** Frozen LLM (e.g., Qwen-2.5-7B (base)).

- **Twist Head:** Linear(Hidden_Dim, 1). Output is scalar value (logit bias).

- **Critic Head:** Linear(Hidden_Dim, 1). Output is token-level Bernoulli probability for masking.

### 5.2. Verifiers ($\phi$)

- **Math:** Python sandbox execution (for code-based math) or symbolic equality.

- **Code:** Unit test pass rate (0.0 to 1.0).

### 5.3. Compute Schedule

- **Warming Up:** Initial generation uses low $\beta$ to maximize coverage.

- **Cooling Down:** As $V_\gamma$ becomes accurate, we shift $\beta \to 1$ to rely more on the learned twist and less on random search.

## 6. Proposed Experiments

### 6.1 Benchmarks

We focus on mathematical reasoning tasks where "lookahead" is critical:

* **Mathematical Reasoning (GSM8K, MATH, MATH500):** Where an error in step 1 invalidates the final answer.

* **HumanEval:** Programming tasks.

* **GPQA, GPQA Diamond:** Science multiple choice questions.



### 6.2 Baselines

#### **Best-of-N brute force rejection sampling:** Defines the difficulty of the problem for the base model. Do a sweep of the first 100 problems at different temperatures [0.6, 0.8, 1.0, 1.2] before taking the best temperature and evaluating all problems at this temperature.

#### **PPO and GRPO:** KL penalty $\beta=0.01 \dots 0.05$.

#### **Standard SMC Steering:** Using a process reward model. Similar to Puri et al. 
**Status:** ✅ Implemented in `trex/baselines/smc_steering_baseline.py` using `Qwen/Qwen2.5-Math-PRM-7B`.

Baseline Implementation: Rollout Roulette (Puri et al., 2025) This baseline uses Sequential Monte Carlo (SMC) (specifically Particle Filtering) to perform "inference-time compute scaling." Instead of a single beam search or independent sampling, it maintains a population of partial solutions, scores them, and re-allocates compute to the most promising ones. Below is the precise algorithm and implementation details.

##### 1. Core Components
- **State Space:** The "particle" is a sequence of tokens. In the paper, the "step" size is typically a full thought step (e.g., a line of reasoning or a paragraph ending in a newline), rather than a single token.

- **Population Size ($N$):** They use a fixed number of particles (e.g., $N=4$ to $N=16$ for efficiency comparisons, scaling up to larger numbers like 128 for max performance).

- **Reward Model ($R$):** A Process Reward Model (PRM) or a Verifier that returns a scalar score $v \in [0, 1]$ for a partial sequence.

##### 2. The Algorithm (Step-by-Step)

Phase 1: Initialization

Start with $N$ identical particles containing just the problem prompt $x$.
$$S_0 = \{x^{(1)}, x^{(2)}, \dots, x^{(N)}\}$$ 
Initialize weights $w^{(i)}_0 = 1/N$.

Phase 2: The Loop (SMC)

Repeat until all particles reach a terminal state (e.g., \boxed{answer} or max tokens).

**1. Expansion (Rollout Step):** For each particle $x^{(i)}_{t-1}$, sample the next "step" using the base LLM $p_\theta$.
 * *Implementation Note:* Generate tokens until a delimiter (e.g., \n or specific step separator) is reached.
 $$x^{(i)}_t \sim p_\theta(\cdot | x^{(i)}_{t-1})$$

**2. Weighting (Scoring):** Score the newly expanded particle using the Reward Model.
$$w_t = w_{t-1} × PRM(step_t)$$
(Crucially, they often use the raw PRM probability or a learned value estimate.)

**3. Resampling (The "Roulette"):** This is the critical step. You select which particles survive to the next round based on their weights.
* Normalize weights: $\tilde{w}^{(i)}_t = \frac{e^{w^{(i)}_t / \tau}}{\sum_j e^{w^{(j)}_t / \tau}}$ (Softmax with temperature $\tau$ is common, or just linear normalization).
* Resampling Strategy: Sample $N$ indices from the distribution defined by $\tilde{w}$ (e.g., Systematic or Multinomial resampling). 
* Replace the current population $S_t$ with the resampled population. High-scoring particles are duplicated; low-scoring ones are dropped.

Phase 3: Final Selection

Once all particles are finished, select the final answer.
- **Majority Voting:** Most common final answer among the particles.
- **Best-of-N:** The particle with the highest ORM score.

#### **Twisted SMC (Ours, No Parallel Temperatures):** Learning the twist with a single swarm at one temperature. Tests the value of Parallel Exploration. Based on Feng et al.

This method replaces standard LLM generation (or Beam Search) with a Twisted Sequential Monte Carlo (TSMC) sampler.

The Core Idea: Instead of hoping the base model stumbles upon the right answer, we train a Value Function (Twist) that predicts whether a current partial reasoning trace will lead to a correct answer.

The Mechanism: This Value Function is used to bias the resampling step of a Particle Filter. We maintain a population of "thoughts." At every step (e.g., every newline), we kill off thoughts with low value estimates and clone thoughts with high value estimates.

##### 1. System Components

You need three specific modules to implement this:

**A. The Base LLM ($p_\theta$)**

- **Role:** The proposal generator. It generates the next few tokens (a "step") given the history.
- **State:** Frozen (usually).
- **Example:** Qwen-2.5-7B (base).

**B. The Twist Function / Value Head ($\psi_\phi$)**

- **Role:** A scalar regression head that estimates $V(s) = P(\text{correct} | s)$.
- **Architecture:** A simple linear layer (Projector) attached to the final hidden state of the Base LLM.
- **Input:** [Batch, Seq_Len, Hidden_Dim]
- **Output:** [Batch, Seq_Len, 1] passed through a Sigmoid.
- **Definition:** $\psi(x_{1:t}) \approx \mathbb{E}_{p_\theta}[R | x_{1:t}]$ where $R \in \{0, 1\}$.

**C. The Verifier / Environment**

- **Role:** Checks if the final answer is correct.

- **Implementation:** A Python function that extracts \boxed{answer} and compares it to the ground truth.

##### 2. The Training Phase (Learning the Twist)

Feng et al. avoid expensive human labeling by using Outcome Supervision to synthesize a dataset. This is effectively "Monte Carlo Policy Evaluation."

Algorithm: Iterative Twist Learning

1. Data Collection (Rollouts)

For a set of math prompts $x \in \mathcal{D}_{train}$:

Sample $K$ full trajectories using the current policy (initially just the Base LLM).
$$\tau^{(k)} \sim p_\theta(\cdot | x)$$
Verify the final answer of each trajectory $\tau^{(k)}$.If correct: Assign Reward $R = 1$.
If incorrect: Assign Reward $R = 0$.

2. Label Assignment (The "Monte Carlo" Trick)
Since we don't know which step caused the error, we assign the final reward to every step in the trajectory.
Dataset $\mathcal{D}_{value} = \{ (x_{1:t}^{(k)}, R^{(k)}) \}$ for all steps $t$ in all trajectories $k$.

3. Training Objective (MSE)

Train the Value Head parameters $\phi$ to minimize the Mean Squared Error between the predicted twist and the realized reward.
$$\mathcal{L}(\phi) = \frac{1}{|\mathcal{D}|} \sum_{(s, r) \in \mathcal{D}} || \psi_\phi(s) - r ||^2$$
Note: It is crucial to re-collect data periodically. As the Twist improves, you use it to sample data (see Inference section), creating a virtuous cycle (Self-Improvement).

##### 3. The Inference Phase (Twisted SMC)

This is the runtime algorithm. It uses the learned Twist to steer generation.

Hyperparameters:
* $N$: Number of particles (e.g., 16).
* $T$: Max number of reasoning steps (lines).

Step-by-Step Logic

Initialization:
Create $N$ particles, all initialized to the prompt.$$S_0 = \{x^{(1)}, \dots, x^{(N)}\}$$
Initialize weights $W_0^{(i)} = 1$.
The Loop (for $t = 1$ to $T$):
**1. Selection (Resampling):**
 * If $t > 0$, normalize the weights $W_{t-1}$.
 * Resample $N$ ancestors from the previous population $S_{t-1}$ proportional to weights $W_{t-1}$.
 * Effect: High-value particles are duplicated; low-value particles are dropped.

**2. Mutation (Proposal):**
 * For each particle, generate the next "step" using the Base LLM $p_\theta$.
 * Implementation: Generate until the next \n token.
 $$x_t^{(i)} \sim p_\theta(\cdot | \tilde{x}_{t-1}^{(i)})$$

**3. Weighting (Twisting):** Calculate the "incremental importance weight." 
* In the optimal TSMC framework, the weight $w_t$ accounts for the change in the value function.
$$w_t^{(i)} = \frac{\psi(x_{1:t}^{(i)})}{\psi(x_{1:t-1}^{(i)})}$$
* Intuition: If the Twist value went up (0.5 $\to$ 0.9), this was a great step $\to$ High Weight. If it went down (0.5 $\to$ 0.1), this was a bad step $\to$ Low Weight.
* Note: If using the Base LLM as the proposal, the likelihood ratio $p_\theta / p_\theta$ cancels out, leaving only the ratio of Twists.

**4. Termination:** 
* If a particle generates \boxed{} or typically finishes, mark it as done. At the end, pick the particle with the highest final Twist value $\psi(x_{1:T})$.

* **Twisted SMC with self-distillation:** Compare against Kim et al. [2025].

* **Replica Exchange SMC (Ours, No Twist Learning):** Parallel tempering without the learned proposal. Tests the value of the twisted proposal.