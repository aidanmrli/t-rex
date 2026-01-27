# Implementation Plan for T-REX

**NOTE:** We should always update this plan with our progress once we have implemented something and it works.

**TODO:** First, look through the repository and see what we already have. Then, plan how to implement the rest of the baselines. Finally, plan how to implement the abstract components of our method, and plan the implementation of the abstract components.

We should have three abstract components:

1. Exploration through the usage of high temperatures and non-reversible parallel tempering to move between states
2. Twisted Sequential Monte Carlo (TSMC) to guide a model at a fixed temperature toward high-value regions efficiently while maintaining diversity of a set of particles
3. An abstract transport mechanism to help move particles between different temperature states. In high-dimensional spaces like language, the "Hot" distribution (Creative/Chaotic) and the "Cold" distribution (Rigorous/Narrow) might have almost zero overlap. If you try to swap them directly, the Cold model calculates an acceptance probability of almost zero.

## Twisted Sequential Monte Carlo (TSMC)

Use simple MLP heads for now to train the twist functions.

We should be using GPU parallelization to speed up the TSMC process as much as possible. Assume we can have either 4 or 8 H100 or H200 GPUs at any one time.

## Abstract Transport Mechanism

### Ideas to Try

We should have a plug-and-play system that can select an individual idea out of these ideas and try it.

#### Level 1: The Mathematical Tricks (Cheapest)

**1. Standard Metropolis-Hastings (The Baseline)**
- **Concept:** Just propose the swap. If $P_{cold}(x_{hot})$ is high enough, accept it.
- **Mechanism:** $A = \min(1, \frac{P_{cold}(x_{hot}) P_{hot}(x_{cold})}{P_{hot}(x_{hot}) P_{cold}(x_{cold})})$
- **Why it fails:** In high dimensions, probabilities multiply. If even one token is "garbage," the whole ratio collapses to zero. Acceptance Rate $\approx 0\%$.

**2. Quantile Coupling (Common Random Numbers)**
- **Concept:** Don't transport the text; transport the "luck." If the Hot model picked its 99th percentile token (a rare word), force the Cold model to pick its 99th percentile token.
- **Mechanism:** Save the CDF values ($u \in [0,1]$) from the hot generation. Map them to the cold inverse CDF.
- **Why it fails:** "Tail Mismatch." The 99th percentile of a Hot model might be a "creative insight." The 99th percentile of a Cold model is usually "gibberish." You map creativity to garbage.

#### Level 2: The Heuristic Bridges (Mid-Cost)

**3. Likelihood Thresholding (Self-Correction)**
- **Concept:** The Cold model looks at the Hot sample and says, "I like most of this, but these 5 words look weird." It masks those words and regenerates them.
- **Mechanism:** Calculate token probabilities under $\pi_{cold}$. If $P(t) < \text{Threshold}$, mask and resample using Block-Gibbs.
- **Why it fails:** "The Conformity Filter." It cannot distinguish between a syntax error (which should be fixed) and a brilliant novel idea (which is also unlikely). It deletes the very novelty you wanted to find.

**4. Best-of-N Rejection (Selection)**
- **Concept:** Instead of fixing one sample, generate 100 Hot samples. Assume one of them effectively bridges the gap by pure chance.
- **Mechanism:** Sample $N$ trajectories. Score them. Pick the best one to propose to the Cold chain.
- **Why it fails:** Inefficient. It requires massive compute to find one "lucky" sample that naturally fits the Cold distribution. If the distributions are too far apart, $N$ must be astronomically large.

#### Level 3: The Verifier Bridges (High-Cost / Engineering SOTA)

**5. PRM-Guided Repair (Process Reward Models)**
- **Concept:** Use an external "Judge" (a math/reasoning verifier) to identify the specific logical step that is broken, rather than looking at token probabilities.
- **Mechanism:** The Hot chain generates a solution. A Verifier scores each step. The step with the lowest score is masked and resampled by the Cold model.
- **Tradeoff:** This is the most robust current method. It fails only if the Verifier is biased against new strategies ("The Galileo Fallacy"), but it is far better than Likelihood Thresholding.

#### Level 4: The Learned Bridges (Highest Complexity / Research SOTA)

**6. Learned Critic + Block-Gibbs Editor (Your T-REX Proposal)**
- **Concept:** Train a specific neural network to recognize "what makes a Hot sample unacceptable to the Cold model" and fix only that part.
- **Mechanism:**
    - **Critic:** Predicts a mask $M$ covering tokens that cause rejection.
    - **Editor:** The Cold model resamples $x_M$ conditional on $x_{-M}$.
    - **Twist:** Updates the proposal distribution to avoid these errors in the future.
- **Why it works:** It is an Active Transport Map. It doesn't rely on luck (Level 1), conformity (Level 2), or a static verifier (Level 3). It learns the optimal way to translate "Hot ideas" into "Cold proofs."
- **Cost:** Requires training a Critic online (or offline) and running sequential Block-Gibbs inference.


