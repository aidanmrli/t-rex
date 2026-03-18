Yes — for one fixed target dataset, I would train it as a dataset-conditional prefix viability verifier, not as a generic off-the-shelf PRM.

The core object should be

C_\phi(x,b_{1:m}) \approx \Pr(\text{verified success on this dataset}\mid x,b_{1:m}),

where x is a problem from your dataset and b_{1:m} is a completed block prefix under your exact blockization. This matches the T-REX requirement that the online reward be a strictly positive, cross-depth calibrated prefix potential that reflects whether a prefix remains plausibly completable, rather than merely local step correctness.

What changes when you say “for the legit target (one particular dataset)” is that you should stop optimizing for generic process-verification quality first, and instead optimize for the dataset-conditional quantity you actually need:

C^\mathcal{D}(x,b_{1:m})
=

\Pr_{\mu_{\text{mix}}^\mathcal{D}}\!\big(\text{terminally verified correct completion on dataset }\mathcal{D}\mid x,b_{1:m}\big).

That is a value-like object, not a pure PRM. Zhang et al. explicitly distinguish these: PRMs are about current-step correctness, while MC future-success labels are really training a value model for future solution potential.

What to train

Train a model with two heads.

The deployed head is:

C_\phi(x,b_{1:m}) \in (0,1),

interpreted as dataset-conditional prefix viability / continuation success probability.

The auxiliary head is:

S_\phi(x,b_{1:m}) \in (0,1),

interpreted as the probability that the newest block introduced a fatal or invalid step.

Why two heads:
 • T-REX wants C_\phi-style semantics for sequential weighting.
 • Zhang et al. show that MC future-success labels are poor substitutes for true step-correctness labels, so step soundness should be learned from a different signal.

Use C_\phi online. Use S_\phi for diagnostics, filtering, and later ablations.

How to construct labels for one dataset

For each problem x \in \mathcal{D}:

 1. Generate many full trajectories using the exact proposer family, decoding config, and blockization you expect online.
 2. Split each trajectory into completed block prefixes b_{1:1},\dots,b_{1:M}.
 3. For selected prefixes, estimate continuation success by conditional rollouts from that prefix.
 4. Verify final answers using the dataset’s strongest available terminal checker.
 5. Store the success count s out of K rollouts for that prefix.

This gives the soft target

\hat C(x,b_{1:m}) = \frac{s}{K}.

That is exactly the kind of supervision the T-REX note recommends: binary eventual-correctness, soft eventual-success estimates, and Monte Carlo reward-to-go from sampled suffix completions, all aligned to the same completed-block segmentation used online.  ￼

Which rollouts to use

Do not define continuation success using only one completion policy.

Instead, use a mixture:

\mu_{\text{mix}}^\mathcal{D} = \sum_{r=1}^R \lambda_r \mu_r.

For a first serious version on one dataset, I would use three sources:

\mu_{\text{mix}}^\mathcal{D}
=

0.5\,\mu_{\text{base}}
+
0.3\,\mu_{\text{hot}}
+
0.2\,\mu_{\text{search}}.

Where:
 • \mu_{\text{base}}: plain ancestral continuation from the frozen SFT/base model,
 • \mu_{\text{hot}}: hotter continuation or softened chain continuation,
 • \mu_{\text{search}}: later, continuation distribution induced by your current sampler.

Why this matters: if you only use the base model, you train “can the base model finish this?” rather than “is this prefix in the viable corridor for this dataset?” T-REX explicitly suggests training from a mixture of ancestral samples, demonstrations, and improved trajectories found by search or SMC.  ￼

What data you actually need

For one dataset, I would build four disjoint splits.

1. Prefix-train

Main training prefixes for C_\phi.

Include:
 • prefixes from base ancestral samples,
 • prefixes from successful traces,
 • prefixes from near-miss traces,
 • prefixes visited by hot/search runs.

1. Prefix-calibration

Held-out prefixes only for post-hoc calibration.

For each prefix here, spend more rollouts to get a better estimate of empirical continuation success.

1. Prefix-switch-eval

Held-out prefixes for model-selection and ablations.

1. Process-aux

A smaller set with explicit local-step labels for the S_\phi head.

Zhang et al. strongly support keeping step-verification supervision separate from MC future-success supervision, and they report gains from combining MC with LLM-as-a-judge via filtering rather than treating MC as the sole truth source.

How to choose prefixes to label

Do not spend equal budget on every prefix.

Prioritize:
 • prefixes near current decision thresholds,
 • prefixes with large sibling disagreement,
 • prefixes from hard problems in the target dataset,
 • prefixes that are plausible but often fail,
 • prefixes where one model says “high viability” and another says “fatal error.”

The reason is simple: your target distribution only cares about prefixes where the viability geometry is hard. T-REX warns that noisy reward ratios cause degeneracy, so your highest-quality labels should be concentrated where the sampler is sensitive.

What objective to train

Use the continuation counts directly.

If a prefix has s_i successful completions out of K_i, train with a Binomial loss:

s_i \sim \mathrm{Binomial}(K_i, C_\phi(x_i,b_{1:m_i})).

That is better than training on just the mean s_i/K_i, because it respects uncertainty from different rollout counts.

The loss is

\mathcal{L}_{\text{viability}}
=

-\sum_i w_i \log \Pr\big(s_i \mid K_i, C_\phi(x_i,b_{1:m_i})\big).

Add a smaller auxiliary loss for the soundness head:

\mathcal{L}
=

\mathcal{L}_{\text{viability}}
+
\lambda_{\text{aux}} \mathcal{L}_{\text{soundness}}.

Keep \lambda_{\text{aux}} small enough that the model remains primarily a viability estimator.

How to get the auxiliary soundness labels

For the target dataset, use a stronger judge or auditor on a subset of prefixes to label the newest block as one of:
 • clean/on-track,
 • recoverable issue,
 • unsupported leap,
 • contradiction/fatal error.

This follows Zhang et al.’s main lesson: if you want step-level process quality, use direct judging/human-like supervision rather than naive MC future-success labels.

Then use those labels to:
 • train S_\phi,
 • audit false positives of C_\phi,
 • mine hard examples.

How to parameterize the deployed reward

Once C_\phi is trained and calibrated, define

R_\phi(x,b_{1:m}) = \varepsilon + \mathrm{Cal}\!\big(C_\phi(x,b_{1:m})\big),
\qquad \varepsilon>0.

Then T-REX uses

w_m = \frac{R_\phi(x,b_{1:m})}{R_\phi(x,b_{1:m-1})},

or at temperature \beta_k,

\log w_m^{(k)}
=

\beta_k
\left[
\log R_\phi(x,b_{1:m})
-

\log R_\phi(x,b_{1:m-1})
\right].

This is exactly the sequential weighting contract in the T-REX note, and it is why cross-depth calibration matters so much.

How to make it place high mass on correct answers for one dataset

There are three levers.

1. Match the label distribution to the dataset

All training prefixes and continuation estimates should come from the actual dataset or a very close superset. If your dataset has special proof style, notation, or difficulty, your verifier should see that distribution directly.

1. Oversample near-correct and boundary prefixes

To shape the narrow corridor, you need many examples of:
 • almost-correct but failing prefixes,
 • recoverable prefixes,
 • subtly doomed prefixes,
 • alternative valid reasoning styles that still solve the problem.

This is where the mass placement is learned.

1. Calibrate on the target dataset, not globally

Fit calibration on held-out prefixes from that same dataset. T-REX is explicit that ranking-trained rewards are insufficient; you need calibration across prefix lengths because the online sampler uses reward ratios.

What I would not do

I would not do any of these as the main plan:
 • Train only on inherited terminal outcomes from one sampled trajectory.
 • Train only on off-the-shelf PRM labels.
 • Treat MC future-success labels as if they were true step-correctness labels.
 • Optimize only Best-of-N style reranking metrics.

Zhang et al. show why this is dangerous: MC supervision induces value-model semantics, not true process verification, and BoN can bias PRMs toward outcome-heavy behavior that hides process weaknesses.

The concrete minimal recipe

For one target dataset, my recommended first serious pipeline is:

 1. Freeze one proposer checkpoint, one decoding config, one blockization.
 2. Generate 5k–20k full traces on that dataset.
 3. Extract all completed block prefixes.
 4. Choose a diverse subset of prefixes for continuation labeling.
 5. For each chosen prefix, run K=4 to 16 continuations under a mixture of rollout policies.
 6. Verify final answers and store (s,K).
 7. On a smaller subset, collect explicit local-step soundness labels.
 8. Train a two-head verifier with Binomial viability loss plus auxiliary soundness loss.
 9. Calibrate C_\phi on a held-out prefix-calibration split.
 10. Deploy R=\varepsilon+\mathrm{Cal}(C_\phi) in SMC and track increment volatility, ESS, and final accuracy.

That is the cleanest way to get a verifier that places high probability mass on correct answers for your actual target dataset, while staying aligned with the T-REX target and with the distinction Zhang et al. draw between value models and true process verifiers.

The most important sentence is this:

Train the online soft verifier as a dataset-matched, calibrated prefix viability model from conditional continuation-success counts, and keep step-soundness as a separate auxiliary signal.
