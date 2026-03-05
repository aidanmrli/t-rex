Here is a model-agnostic recipe to reproduce the Twisted Sequential Monte Carlo (TSMC) method for math reasoning, based on Feng et al. (2024).

# 🧪 Recipe: Twisted Sequential Monte Carlo (TSMC) for Math

This recipe outlines the three-stage pipeline required to reproduce the results. It is designed to be model-agnostic, meaning you can swap the underlying LLM (e.g., Llama, Mistral, Qwen) as long as it supports fine-tuning.

## 📋 Prerequisites

* **Base Model:** A pre-trained base LLM (not Instruction-tuned).
* **Dataset:** A math dataset containing step-by-step solutions (e.g., **PRM800K** or **MATH** with Chain-of-Thought traces).
* **Compute:** Sufficient GPU memory to load two copies of the model (Generator + Value Function) during inference, or ability to switch adapters.

---

## Stage 0: PRM800K Download + Explore (Required)

Before any formatting or training, **download and explore PRM800K** to lock in the schema and verify that the data includes stepwise reasoning. This prevents silent schema drift and ensures we format steps correctly.

**Command (explore only, Hugging Face):**
```bash
python trex/prepare_math_datasets.py \
  --include_prm800k \
  --only_prm800k \
  --prm800k_stage explore \
  --prm800k_source hf \
  --prm800k_dataset tasksource/PRM800K
```

If you hit Arrow casting errors from `datasets`, retry with streaming:
```bash
python trex/prepare_math_datasets.py \
  --include_prm800k \
  --only_prm800k \
  --prm800k_stage explore \
  --prm800k_source hf \
  --prm800k_dataset tasksource/PRM800K \
  --prm800k_streaming
```

**Artifacts:**
- `trex/data/prm800k_schema.json`
- `trex/data/prm800k_samples.jsonl`

Optional: set `--prm800k_config <config_name>` if the dataset exposes multiple configs.

---

## Stage 1: Supervised Fine-Tuning (Generator)

The goal is to teach a base model to reason step-by-step and, crucially, to use a consistent **delimiter** between steps. This delimiter acts as the "synchronization point" for the Monte Carlo search.

### 1. Data Formatting (PRM800K)

Format PRM800K so that **every reasoning step ends with `\n\n`**. The delimiter is **learned purely from data formatting** (no prompt instruction).

**Preferred field:** `question.pre_generated_steps` (list of step strings).  
**Fallback:** `question.ground_truth_solution` normalized to `\n\n` between steps/paragraphs.

**Command (format only, Hugging Face):**
```bash
python trex/prepare_math_datasets.py \
  --include_prm800k \
  --only_prm800k \
  --prm800k_stage format \
  --prm800k_source hf \
  --prm800k_dataset tasksource/PRM800K \
  --prm800k_use_pre_generated_steps \
  --prm800k_filter_correct \
  --prm800k_output trex/data/prm800k_sft_train.jsonl
```

### 2. Training Configuration

* **Model:** Base LLM (e.g., 7B parameter size).
* **Loss:** Standard autoregressive Cross-Entropy loss.
* **Hyperparameters:**
* Epochs: 1–3 (until convergence on validation syntax).
* Learning Rate: Standard SFT rates (e.g., ).


* **Outcome:** A **Generator Model ()** that reliably outputs `\n\n` after distinct reasoning steps.

---

## Stage 2: Value Function Learning (The "Twist")

The core of TSMC is a learned "Twist" function (Value Function) that predicts the probability of the final answer being correct given a partial solution.

### 1. Data Generation (Self-Sampling)

Use your **Generator ()** from Stage 1 to generate solutions for your training set problems.

* **Sampling:** Generate  solutions per problem (e.g., ).
* **Labeling:** Compare the final generated answer to the ground truth.
* Mark trajectory as **Positive (1)** if correct.
* Mark trajectory as **Negative (0)** if incorrect.



### 2. Value Function Training

Initialize a new model from the **Generator** weights and add a scalar regression head (linear layer) on top of the last hidden state.

* **Objective:** **Contrastive Twist Learning (CTL)**.
* Instead of standard MSE, minimize the KL divergence between the target distribution and the twisted distribution.
* *Simplification:* You can often approximate this by training the value head to predict the binary correctness label (0 or 1) using a Binary Cross Entropy (BCE) loss on the generated step data.


* **Target Token:** Train the model to predict this value specifically at the `\n\n` delimiter tokens.
* **Outcome:** A **Value Function ()** where  represents the probability that partial sequence  leads to a correct answer.

---

## Stage 3: TSMC Inference

At test time, you alternate between **Generation** and **Resampling**.

### Algorithm Parameters

* **Particles ():** Number of parallel sequences (e.g.,  or ).
* **Warm-up:** Skip resampling for the first  tokens (e.g., ) to allow initial context buildup.
* **Max Resampling Steps:** Limit to roughly 5-10 times per problem to save compute.

### The Inference Loop

For each test problem :

1. **Initialize:** Start with  identical particles containing .
2. **Generate:** Run the **Generator** on all  particles until they all reach the next `\n\n` delimiter (or end of generation).
3. **Calculate Weights:**
For each particle , calculate the incremental importance weight .
The paper defines the optimal twist weight proportional to the square root of the value function:



*(Note: If  is constant across particles, you only need .)*
4. **Resample:**
* Normalize weights: .
* Resample  new particles from the current pool based on .
* *Recommendation:* Use **Stratified Sampling** to reduce variance (prevents dropping good unique paths too aggressively).


5. **Repeat:** Continue generating from the new set of particles until the EOS token is reached.
6. **Final Selection:** Vote among the final  completed solutions to select the answer.

---

## ⚠️ Implementation Pitfalls to Avoid

1. **Tokenizer Issues:** Ensure your tokenizer does not merge `\n\n` with other words. The logic must strictly trigger *exactly* when the model outputs the specific token ID for the second newline.
2. **Weight Degeneracy:** If your Value Function is too confident (predicts 0.0001 vs 0.99), resampling will kill all diversity instantly.
* *Fix:* Apply a temperature or smoothing to the weights before resampling.


3. **Observation Mismatch:** Ensure the Value Function sees the *exact* same string context as the Generator. A missing space or newline difference will ruin the predictions.
