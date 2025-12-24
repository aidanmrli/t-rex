Here is the content of the provided PDF formatted in Markdown.

T-REX: Twisted Replica Exchange for Bootstrapping Reasoning Guidance in LLMS 

**Author:** Aidan Li 

### Abstract

Reasoning with hard constraints creates a "narrow passage" problem for large language models (LLMs). Existing population-based Monte Carlo methods face a fundamental dilemma: Twisted Sequential Monte Carlo (TSMC) requires valid trajectories to learn twist guidance, while Parallel Tempering suffers from inefficient random walks in high-dimensional discrete spaces.

We introduce **Twisted Replica Exchange (T-REX)**, a framework that unifies these paradigms to bootstrap reasoning capabilities from scratch. T-REX leverages a key insight: in reasoning tasks, high-temperature sampling yields semantic diversity (e.g., alternative problem-solving strategies) rather than incoherent noise. Our method utilizes these hot chains to explore the solution landscape and discover sparse "proto-solutions". These trajectories are used to train a twist function online, which progressively biases the proposal distribution of the cold chains toward valid regions. To ensure efficient transport between modes, we employ a sparsity-constrained Block-Gibbs editor that performs local repairs on invalid tokens. Empirical results on complex reasoning benchmarks (e.g., GSM8K) demonstrate that T-REX effectively learns to search during inference, achieving higher success rates than standard sampling while utilizing adaptive compute to resolve hard constraints.

---

## 1. Introduction

The **"Narrow Passage"** problem in probabilistic inference describes the challenge of sampling from a distribution where the probability mass is concentrated in a tiny, isolated region of the state space.
In the context of Large Language Models (LLMs), this phenomenon is acute during complex reasoning tasks: a single logical error or arithmetic mistake in an early step renders the entire subsequent trajectory invalid, regardless of its fluency. While LLMs excel at intuitive generation, they struggle with these hard constraints because standard autoregressive sampling is myopic; it cannot see the future constraint violation until it has already occurred.

To address this, recent work has framed inference-time scaling as a probabilistic sampling problem, utilizing two distinct classes of methods:

1. 
**Exploration-Based (Parallel Tempering):** These methods run parallel chains at varying temperatures to flatten the energy landscape, allowing the model to traverse barriers between modes. However, in the high-dimensional discrete state space of text, heating a chain often results in a diffusive random walk. Without direction, the model wastes compute exploring incoherent regions rather than semantically diverse reasoning paths.


2. 
**Guidance-Based (Twisted SMC):** These methods learn a "twist" (or value) function to bias the token proposal distribution toward high-value regions, effectively steering the model through the narrow passage. However, these methods face a fundamental **"chicken-and-egg" paradox**: to learn an accurate twist function, one needs a dataset of valid solution trajectories; yet, these valid trajectories are initially unknown and we have no method to sample them efficiently.



We propose **T-REX**, a framework that unifies these approaches to resolve their respective limitations. We leverage the diversity of Replica Exchange to break the "chicken-and-egg" cycle of Twisted SMC.

### Key Insight

Our key insight is that in reasoning tasks, heating the model  does not produce random noise, but rather semantic diversity. It allows the hot chains to brainstorm alternative reasoning strategies (e.g., algebraic vs. heuristic) via the model's strong prior. These hot chains discover the initial sparse solutions required to train the twist function online. In turn, the learned twist guides the cold chains, progressively narrowing the search space around valid solutions.

This approach establishes a new paradigm for **Adaptive Test-Time Compute**. Rather than treating the proposal distribution as static, we demonstrate that inference compute should be allocated dynamically: not just to search for solutions, but to learn *how* to search in real-time.

---

## 2. Methodology

### 2.1 Problem Formulation

We address the problem of generating sequences  from a Large Language Model (LLM)  subject to strict, non-local constraints . The target distribution is the posterior:



In standard LLM sampling, this poses a Narrow Passage problem. The set of valid reasoning trajectories constitutes a sparse subspace within the vast combinatorial output space of the prior . Mathematical reasoning is logically fragile: a single arithmetic or deductive error early in the chain irreversibly steers the generation off the valid path. Standard autoregressive sampling is myopic to these long-horizon constraints.

### 2.2 Why Tempering? (Smoothing the Landscape)

To traverse this challenging state space, we employ **Parallel Tempering (PT)**. The intuition is to melt the strict constraints into a smooth energy landscape.  We construct a sequence of  distributions  defined by an inverse temperature schedule :



* **At  (Hot):** The landscape is flat. Barriers are removed, allowing hot particles to traverse the space freely and discover diverse (though potentially invalid) proto-solutions.


* **At  (Cold):** The landscape is rugged. Cold particles are locked into valid regions but cannot cross energy barriers to find better modes.



By exchanging states between these temperatures, we allow cold chains to escape local optima by teleporting to a hotter state, moving across the barrier, and cooling back down in a new valid mode.

### 2.3 Global Communication: Non-Reversible (Discrete) Neural Transport

Standard PT fails in high dimensions for two reasons:

1. Random walk behavior makes traversing the temperature ladder slow ( steps).


2. Lack of distributional overlap makes swap acceptance rates vanish.



We address these by combining Non-Reversible Parallel Tempering (NRPT) with a Discrete Neural Transport mechanism based on Block-Gibbs.

#### 2.3.1 The Schedule: Non-Reversible PT (NRPT)

To solve the random walk problem, we use a deterministic, alternating schedule:



By toggling between these sets, we induce a ballistic flow, allowing a particle to propagate from prior to posterior in  steps.

#### 2.3.2 The Mechanism: Adaptive Block-Gibbs Transport

Continuous normalizing flows are inapplicable to discrete token sequences. We introduce **Adaptive Block-Gibbs Transport**. Instead of learning a global transport map, we learn a local Discrete Editor that modifies a "hot" particle  to increase its likelihood of acceptance in the "cold" distribution .

**Phase 1: The Learned Masking Policy**
We employ a lightweight Critic network , parameterized as a token-level binary classifier. Given a trajectory  and target temperature , the Critic predicts a binary mask  identifying tokens responsible for constraint violations.


We impose a hard sparsity constraint  (where ) to ensure edits remain local and efficient.

**Phase 2: Base Model In-filling**
We freeze the unmasked context  and resample the masked tokens using the base pre-trained LLM :


Crucially, we use the base model  rather than the twisted proposal, ensuring the proposal kernel matches the prior.

**The Transport Acceptance Ratio**
We treat the edit  as a Metropolis-within-Gibbs proposal. The acceptance probability  is:



The complex likelihoods  cancel out, reducing to:




### 2.4 Local Guidance: The Twisted Proposal

To improve exploration within a distribution, we introduce a Twist Function .  Its role is to estimate the log-probability that the current prefix can be completed into a valid solution. The proposal distribution is twisted to favor promising prefixes:



#### 2.4.1 The Feedback Loop

 is trained online using the output of the Parallel Tempering explorers.

1. 
**Exploration:** Hot swarms find rare valid paths.


2. 
**Learning:** We compute target value  and update  to minimize MSE: .


3. 
**Exploitation:** The updated twist is immediately used to guide cold swarms.



---

## Algorithm 1: T-REX

**Require:** LLM , Temperatures , Constraints 
**Initialize:** Particles , Twist , Mask Critic 

**For  to  do:**

1. **Twisted Local Exploration (Frontier Expansion)**
* For all particles  in all swarms :
* Sample 


* Calculate weights  and Resample within swarms.


2. **Global Communication (NRPT + Block-Gibbs)**
* Select swap pairs  via deterministic even/odd schedule.
* For :
* Let  (Hot particle).
* **a. Critic:** Sample mask .
* **b. In-fill:** Generate repair  using Base LLM.
* **c. Accept:** Calculate  (Eq. 6).
* If :
* Swap: , 






3. **Online Learning Loop**
* Identify valid/promising trajectories .
* Update Twist : Minimize MSE on .
* Update Critic : Maximize likelihood of masks that were accepted ().



**End For**



---

## 3. Related Work

### 3.1 Inference-Time Scaling via Probabilistic Inference

Recent research has re-framed inference-time scaling from a search problem to a formal probabilistic inference task.

* 
**Diffusion:** Dou and Song [2024] formulated inverse problem solving as non-linear filtering.


* **LLMs:** Puri et al. [2025] modeled text generation as a state-space model, showing Particle Filtering outperforms Beam Search.


* 
**Limitation:** These approaches rely on the pre-trained model for proposals, leaving them vulnerable to the narrow passage problem where the prior rarely proposes valid tokens.



### 3.2 Twisted Sequential Monte Carlo

Twisted SMC methods introduce a learned auxiliary function to guide proposals.

* Feng et al. [2025] and Zhao et al. [2024] apply this to LLMs .


* 
**The Catch-22:** Learning an accurate twist requires high-quality valid trajectories, which are initially unknown.


* Kim et al. [2025]: Addressed this via Self-Distilled Twisted SMC, but this relies on expensive, multi-stage iterative retraining .



### 3.3 Parallel Tempering and Replica Exchange

* 
**Continuous Sampling:** Accelerated Parallel Tempering (APT) uses Neural Transports to warp state space between temperatures. CREPE applies Replica Exchange to control diffusion models.


* 
**Sequential Settings:** Shestopaloff and Doucet [2019] used parallel replicas for lookahead information. Nabika et al. [2025] proposed Sequential Exchange Monte Carlo (SEMC) to automate tuning.



### 3.4 Differentiation and Contribution

T-REX performs **test time training**; it learns during inference for a specific problem. Every new user query is a mini-RL problem where the model must explore, gather data, train a guide, and generate the solution.

---

## 4. Proposed Experiments

### 4.1 Benchmarks

We focus on mathematical reasoning tasks where "lookahead" is critical:

* 
**Mathematical Reasoning (GSM8K, MATH, MATH500):** Where an error in step 1 invalidates the final answer.


* 
**HumanEval:** Programming tasks.


* 
**GPQA, GPQA Diamond:** Science multiple choice questions.



### 4.2 Baselines

* 
**Best-of-N brute force rejection sampling:** Defines the difficulty of the problem for the base model. Do a sweep of the first 100 problems at different temperatures [0.6, 0.8, 1.0, 1.2] before taking the best temperature and evaluating all problems at this temperature.

* 
**PPO and GRPO:** KL penalty $\beta=0.01 \dots 0.05$.


* **Standard SMC Steering:** 


* **Twisted SMC (Ours, No Parallel Temperatures):** Learning the twist with a single swarm. Tests the value of Parallel Exploration.


* **Twisted SMC with self-distillation:** Compare against Kim et al. [2025].


* **Replica Exchange SMC (Ours, No Twist Learning):** Parallel tempering without the learned proposal. Tests the value of the twisted proposal.
