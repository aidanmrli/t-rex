Do not start by training the soft verifier first.

Start by building the minimal data-and-sampling loop that can produce prefixes, continuations, and a few correct or near-correct trajectories. Then train the first soft verifier on that data. Only after that should you try the full multi-chain / parallel-tempering-style sampler.

That order follows directly from the T-REX note: the reward model is critical, but its training data should come from a mixture of sources including ancestral samples, demonstrations, and improved trajectories found by search or SMC. It also says the natural supervision targets include binary eventual correctness, soft eventual-success estimates, and Monte Carlo reward-to-go from sampled suffix completions. In other words, the verifier needs a data pipeline first; it should not be the very first thing you build.

Here is the order I would use.

1. Freeze the problem setup first

Pick:
 • one dataset,
 • one base model,
 • one prompt format,
 • one blockization scheme,
 • one terminal checker.

You need this frozen before serious verifier work, because T-REX’s reward must be calibrated on the same completed-block representation used online, and mismatch in segmentation or rollout setup can break the incremental ratios R(b_{1:m})/R(b_{1:m-1}).

1. Build the cheapest possible trajectory-generation pipeline

Before any fancy sampler, make sure you can generate:
 • full trajectories,
 • completed block prefixes,
 • terminal verification labels,
 • stored metadata tying each prefix to model, decoding config, and blockization.

At this stage, use only very simple generation:
 • ancestral sampling,
 • temperature/top-p sweeps,
 • maybe best-of-N with a terminal verifier,
 • maybe a tiny amount of teacher-forced or demonstration data if you have it.

This is how you get the first useful trajectories even when the model is weak.

1. Get useful trajectories without already solving many problems

You do not need the base model to solve many problems by itself.

For the first verifier, you only need a dataset of prefixes spanning:
 • obvious failures,
 • near misses,
 • partial progress,
 • occasional successes,
 • and, ideally, some corrected or demonstrated solutions.

That is enough because the soft verifier is supposed to learn viability, not only perfect correctness. The T-REX note explicitly says training should emphasize whether a completed prefix is still extendable to a correct final solution, because a prefix may be locally flawed yet still repairable.  ￼

So the initial useful prefixes can come from:
 • easy subset / curriculum within the dataset,
 • multiple random seeds per problem,
 • higher-temperature decoding to diversify attempts,
 • best-of-N to harvest rare successes,
 • teacher demonstrations / corrected traces if available,
 • partial prefixes from incorrect traces that still contain real progress.

You are not waiting for the model to be good. You are building a noisy map of which prefixes are dead, recoverable, or promising.

1. Train a v0 viability model, not a polished PRM

Your first verifier should be a small, simple, custom model trained on cheap labels.

The main target should be value-like:

C_\phi(x,b_{1:m}) \approx \Pr(\text{verified successful completion}\mid x,b_{1:m}),

not strict step correctness. Zhang et al. explicitly argue that Monte Carlo future-success supervision behaves like a value model, not a true PRM for deterministic current-step correctness. For your use case, that is actually appropriate, because your online object is a soft prefix potential, not a classical process judge.

But I would keep this first version simple:
 • train on inherited outcome labels plus some conditional continuation estimates,
 • calibrate it,
 • use it only in single-chain blockwise SMC first.

Do not jump to multi-temperature chains yet.

1. Run single-chain blockwise SMC before parallel tempering

This is the most important sequencing recommendation.

Before multi-chain tempering or hot-bank injection, verify that:
 • your blockization works,
 • your reward increments are stable,
 • ESS does not collapse immediately,
 • the verifier helps more than it hurts.

T-REX’s own math says the base-model proposal gives incremental weights

w_m = \frac{R(b_{1:m})}{R(b_{1:m-1})},

and warns that if R is noisy or poorly calibrated, these ratios will have high variance and induce degeneracy. That means you should debug the single-chain version first, because it isolates whether your soft verifier is actually usable.

1. Use single-chain runs to improve the verifier data

Once single-chain SMC is working at all, use it to generate better data:
 • prefixes near resampling thresholds,
 • prefixes that the verifier rates highly but still fail,
 • prefixes with rare successful continuations,
 • sibling branches with sharply different outcomes.

This is where the project becomes self-bootstrapping:
 • weak verifier → slightly better search,
 • slightly better search → more informative prefixes,
 • more informative prefixes → better verifier.

That is the right loop.

1. Only then add multi-chain / tempering

After the single-chain setup is stable, then add:
 • temperature ladder,
 • multiple chains,
 • hot-bank injection / cross-chain transfer,
 • and later any twist/proposal experiments.

T-REX’s main novelty is the interacting tempered SMC construction, but the note also makes clear that the whole method depends critically on the quality of the block-prefix reward model. So there is no point debugging cross-temperature communication before you know the basic reward model is usable.

⸻

So what should you do first, concretely?

I would use this exact order:

Phase A: bootstrap data

Generate trajectories from the base model with simple sampling. Harvest:
 • full traces,
 • prefixes,
 • terminal pass/fail labels,
 • a few rare successes,
 • maybe some demonstrations.

Phase B: first soft verifier

Train a small custom viability model on those prefixes. Keep the target simple:
 • inherited final outcomes as weak labels,
 • conditional continuation estimates on a selected subset as stronger labels,
 • optional local soundness labels on a tiny audited subset.

Phase C: single-chain SMC

Run blockwise SMC with that verifier.
Do not do multi-chain yet.

Phase D: iterative improvement

Use prefixes discovered by single-chain SMC to retrain the verifier.

Phase E: multi-chain tempering

Only after the above works, add the parallel-tempering-style mechanism.

⸻

“But what if my model can’t solve the problems?”

Then you should not jump straight into the full target dataset at full difficulty.

Use one or more of these bootstraps:

 1. Curriculum slice
Start with the easiest subset of the dataset.
 2. Rare-success harvesting
Run many cheap samples and keep the few successes. Even a low success rate is enough to seed training.
 3. Demonstrations / solved traces
Add teacher-forced or externally solved trajectories. T-REX explicitly lists corrected solutions from teacher-forced demonstrations as a valid training source.  ￼
 4. Near-success prefixes
Prefixes from failed traces are still useful if some continuations from them can succeed. Those are exactly the kinds of examples a viability model should learn from.
 5. Outcome verifier + continuation rollouts on prefixes
Even when full solves are rare, some prefixes may still show nonzero empirical continuation success under multiple rollouts. That gives you the soft signal you need.

This is why I do not think “the model can’t yet solve the problems” blocks verifier training. It only blocks training a very strong verifier immediately.

⸻

The key principle

You do not need to solve the hard problems first in order to train the soft verifier.

You need to build a pipeline that can produce:
 • diverse prefixes,
 • occasional successes,
 • verified failures,
 • and continuation-based estimates of which prefixes are still alive.

That is enough to train a useful first verifier.

⸻

My recommendation in one sentence

Start with simple generation → first custom viability verifier → single-chain SMC → improved verifier → multi-chain tempering, not verifier-first in isolation and not full parallel tempering first. This order best matches the T-REX dependency structure and avoids building the most complex sampler before you have a reward model that is stable enough to support it.
