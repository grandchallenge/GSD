# GSD-001 — Generalization-State Dynamics Research Programme

**Status:** executable programme plan  
**Date:** 2026-10-04  
**Governing authority:** \`grandchallenge/GSD\`  
**Governing issue:** #139  
**Primary motivating source:** Wen, Wu, Song, Chen, *Generalization Dynamics of LM Pre-training*, arXiv:2609.33150v1  
**Programme type:** empirical + mechanistic + controlled-training research  
**Default execution style:** bounded work packages, exact checkpoint identities, replayable artifacts, independent adversarial lanes where useful

## 1. Mission

Determine whether non-monotone generalization states are:

1. reproducible across checkpoint series;
2. predictable from internal or optimizer-state structure;
3. caused by loss of a computation, changed accessibility/control, or changed behavioural competition;
4. recoverable by bounded interventions;
5. controllable through data/curriculum choice;
6. materially affected by optimizer geometry or architecture.

The programme does not begin from the assumption that the capacity-allocation explanation in arXiv:2609.33150v1 is correct.

The core object is the distinction

\[
\text{acquisition}
\neq
\text{persistence}
\neq
\text{accessibility/control}
\neq
\text{behavioural expression}.
\]

Every work package must preserve that distinction.

## 2. Programme thesis

The central experimental question is not:

> Does benchmark accuracy go up with training?

It is:

> Which computation controls behaviour at each training state, what survives when that control changes, and can training be steered so transferable computations remain accessible and dominant?

A useful structured state description is

\[
\mathcal G_t =
\bigl(
B_t,
M_t,
R_t,
O_t,
C_t
\bigr),
\]

where:

- \(B_t\): behavioural generalization evidence;
- \(M_t\): mechanistic/intervention evidence;
- \(R_t\): representation/operator evidence;
- \(O_t\): optimizer-state/dynamical evidence;
- \(C_t\): control/curriculum context.

These coordinates are observations, not presumed latent ground truth.

## 3. Claim firewall

The programme must not silently promote any of the following:

- a behavioural hop into proof of a circuit replacement;
- a soft-score change into proof of a phase transition;
- a recoverability result into proof that exactly the same mechanism persisted;
- one diagnostic scalar into a universal predictor;
- one probe's controlled behaviour into broad intelligence;
- one successful data-selection intervention into a general curriculum law;
- checkpoint averaging into a principled cure;
- a small-model result into a 32B result;
- an optimizer/architecture difference into a mechanism claim without matched controls;
- a later checkpoint into a better checkpoint merely because it consumed more tokens.

## 4. Model and compute ladder

The programme is intentionally staged by cost.

### Tier A — instrumentation and cheap dynamics

Primary candidates:

1. \`allenai/OLMo-2-0425-1B-early-training\`
   - frequent early-training checkpoints;
   - suitable for evaluator debugging and transition-detection validation;
   - small enough for single-GPU repeated sweeps.

2. \`allenai/OLMo-1B\`
   - full pre-training checkpoint series released at roughly every 1,000 steps;
   - suitable for long-horizon trajectory analysis;
   - official OLMo checkpoint format can include model, optimizer, and trainer state.

Target hardware:

- T4 16 GB for behavioural sweeps;
- L4 24 GB or A100 when retaining many activations;
- CPU/NVMe offload for optimizer-state analysis where necessary.

Tier A is not expected to reproduce every large-model mode hop. Its purpose is to establish a correct and cheap measurement pipeline and determine which probes show usable dynamics at small scale.

### Tier B — primary GCL replication scale

Primary candidate:

- \`allenai/Olmo-3-1025-7B\`.

Reasons:

- same model family as the motivating paper;
- public intermediate checkpoints and training details;
- much cheaper than 32B;
- large enough to test whether discovered signatures survive scale.

Target hardware:

- L4 24 GB for inference/log-probability sweeps with careful memory management;
- A100 preferred for activation-heavy mechanistic diagnostics;
- do not use quantized weights for a claim unless quantization invariance is established first.

### Tier C — targeted high-cost confirmation

Primary candidate:

- OLMo3 32B general-pretraining checkpoint series.

Use only after earlier gates establish a reproducible measurement stack and at least one concrete mechanistic or control hypothesis.

Target hardware:

- A100/H100 80 GB class or multi-GPU tensor parallelism;
- continued pre-training only for narrowly specified interventions.

Tier C is where exact-paper-scale phenomena such as the reported late-training 32B hops, checkpoint selection, and downstream post-training transfer can be tested.

## 5. Evaluation suite

### 5.1 Paper-replication probes

Implement the six behavioural families in arXiv:2609.33150v1:

1. **Flipped Answer** — memorized labels versus in-context task inference.
2. **Repetitive Answer** — copying repeated demonstration answers versus solving the task.
3. **Successive Answer** — following a successor pattern versus solving the task.
4. **Truthy Answer** — truthiness versus truth.
5. **Intuitive Answer** — fast intuitive answer versus reflective solution.
6. **Multi-hop Persona QA** — disconnected facts versus coherent latent persona.

Where the paper exposes multiple underlying datasets, keep each dataset separate before aggregating.

### 5.2 GCL-native probes

Add small probes tied to existing programmes, but never mix them into the paper-replication score without a declared aggregation rule.

Candidates:

- induction versus function inference;
- Dyck / stack-like state tracking;
- entity tracking;
- simple backtracking;
- arithmetic with adversarial in-context distractor patterns;
- relative-position/operator probes where K-DIAGNOSTICS already has a valid interface.

These exist to test transfer of the phenomenon to GCL mechanisms, not to increase the chance of finding a positive result.

### 5.3 Stable controls

At each checkpoint series, run a small stable-control suite covering ordinary capability.

Purpose:

- detect generic checkpoint noise;
- distinguish task-specific generalization-state movement from global degradation;
- catch broken tokenizer, prompt, or log-probability pipelines.

Controls should include at least:

- ordinary sentiment/topic classification without label flipping;
- conventional arithmetic/QA without adversarial answer patterns;
- perplexity or held-out language-model loss where available.

## 6. Primary measurement contract

### 6.1 Soft generalization margin

For each item \(x\) with a predefined generalizing answer span \(a_G\) and shallow/pattern answer span \(a_P\), compute

\[
m_t(x)
=
\log P_{\theta_t}(a_G\mid x)
-
\log P_{\theta_t}(a_P\mid x).
\]

Report:

- per-item margins;
- dataset mean and median;
- bootstrap confidence intervals;
- fraction with positive margin;
- hard accuracy only as a secondary readable summary.

This directly protects against the "metric threshold creates emergence" failure mode.

### 6.2 State labels

Use three labels, not a forced binary:

- \`GENERALIZING\`;
- \`PATTERN_MATCHING\`;
- \`UNCERTAIN\`.

A default dataset-level classification is permitted when the bootstrap interval for the aggregate soft margin lies entirely above or below zero.

Do not hide ambiguous checkpoints by forcing a sign.

### 6.3 Transition definition

A candidate transition requires:

1. a change between confidently classified regimes;
2. checkpoint spacing within the predeclared local window;
3. a statistically resolved soft-margin shift;
4. stable-control performance that does not show a matching global collapse;
5. exact checkpoint identities.

A **confirmed transition** additionally requires either:

- support from flanking checkpoints showing local regime persistence; or
- independent replay of the same checkpoint pair and evaluation.

### 6.4 Cross-probe coherence

Compute a checkpoint-by-probe matrix and pairwise correlation structure.

The objective is to distinguish:

- local probe-specific hopping;
- shared generalization-state movement;
- model-scale dependence of coherence.

Do not average probes into one score until their covariance structure is known.

## 7. Evidence ladder: from behaviour to mechanism

Each transition should move only as far up this ladder as evidence allows.

### Level E0 — behavioural transition

Only the soft behavioural margin has changed.

Disposition:
\`BEHAVIOURAL_STATE_CHANGE\`.

### Level E1 — representation/operator shift

There is a reproducible internal-state or operator change aligned with the transition.

Candidate tools:

- layerwise CKA / Procrustes-aligned representation similarity;
- logit attribution;
- key/query mean and centered geometry;
- RPO/DC/frequency diagnostics;
- attention-head specialization metrics;
- function-vector or task-vector probes.

Disposition:
\`REPRESENTATIONAL_CHANGE_SUPPORTED\`.

### Level E2 — accessibility/control shift

A bounded intervention restores or suppresses the behaviour without substantial retraining.

Candidate interventions:

- prompt/control-token changes;
- activation steering;
- activation patching from an adjacent good checkpoint;
- head/path patching;
- sparse module intervention.

Disposition:
\`ACCESSIBILITY_SHIFT_SUPPORTED\`.

### Level E3 — persistence supported

Evidence indicates a computation remains recoverable or causally usable across the behavioural disappearance.

This may be supported by:

- successful cross-checkpoint activation/path substitution;
- a stable task-relevant subspace that can be reactivated;
- low-cost recovery under an intervention that is too small to plausibly relearn the capability from scratch.

Disposition:
\`MECHANISM_PERSISTENCE_SUPPORTED\`.

This still does not prove literal circuit identity.

### Level E4 — mechanism change supported

Interventions, localization, and reconstruction evidence jointly indicate that the prior computation is no longer functionally present under the declared mechanism class.

Disposition:
\`MECHANISM_CHANGE_SUPPORTED\`.

This is a high bar.

## 8. Recovery-cost programme

Define a nested intervention ladder.

### R0 — context-only recovery

Examples:

- prompt wording;
- order of demonstrations;
- explicit task instruction.

Interpretation:
tests behavioural accessibility, not structural persistence.

### R1 — activation-space recovery

Examples:

- steering vector;
- residual-stream patch;
- function-vector injection.

Interpretation:
stronger evidence for latent accessibility.

### R2 — sparse causal recovery

Examples:

- head patching;
- MLP/block patching;
- low-dimensional operator substitution.

Interpretation:
candidate localized persistence evidence.

### R3 — parameter-light recovery

Examples:

- fixed-rank LoRA;
- sparse parameter edit;
- very small number of gradient steps.

Interpretation:
ambiguous between recovery and rapid relearning; requires controls.

### R4 — continued-training recovery

Examples:

- short controlled pre-training;
- selected-data continuation.

Interpretation:
useful for control, weakest for proving mechanism persistence.

For each tier define cost \(C(\delta)\) before the experiment.

A generic recovery metric is

\[
R_k(\theta,q)
=
\inf_{\delta\in\Delta_k}
C(\delta)
\quad
\text{s.t. target criterion is restored}.
\]

The Residual programme should consume the smallest intervention level that reliably reconstructs the target capability.

## 9. CPS / optimizer-state programme

Public OLMo training checkpoints can include optimizer state. Use that capability where exact checkpoint artifacts are available.

For a transition-local window, capture:

- Adam first moment \(m_t\);
- Adam second moment \(v_t\);
- parameter update direction;
- effective per-coordinate step;
- gradient/update cosine relationships;
- layerwise update norms;
- learning-rate and scheduler state.

Then apply CPS-style augmented-state analysis.

Candidate quantities:

- randomized JVP/VJP estimates of the local augmented-state map;
- dominant singular gain;
- finite-horizon gain;
- spectral-radius estimates where meaningful;
- non-normality proxies;
- angle between leading transient directions and task-gradient directions.

The primary CPS question is predictive:

> Does an optimizer-state signature at checkpoint \(t\) predict a behavioural state change at \(t+\Delta\) on held-out transitions?

Retrospective best-layer selection is exploratory only.

## 10. K-DIAGNOSTICS / operator programme

Transition-local K-DIAGNOSTICS should examine:

- centered and affine key geometry;
- query means;
- RoPE/RPO frequency structure;
- DC components;
- per-head specialization;
- relative-position operator modes;
- head-level structural drift.

The target is not "find any changing metric."

The target is:

> Find a structured internal change that is reproducible across transitions and has a causal or predictive relation to the generalization-state change.

Where a candidate signature appears, pre-register the layer/head/operator statistic and test it on held-out transitions.

## 11. Curriculum-control programme

### 11.1 Data-window attribution

Given a checkpoint transition \(t\rightarrow t+1\), identify the actual data window when accessible.

Estimate which examples or data classes push the target soft margin toward:

- generalization;
- pattern matching;
- no material change.

Methods may include:

- exact micro-update tests;
- gradient similarity to the probe objective;
- influence approximations;
- grouped data-ablation/continuation experiments.

Attribution is not yet control.

### 11.2 Controlled continuation

From a common checkpoint, compare matched-token continuations:

- \`RANDOM\`;
- \`GENERALIZATION_SELECTED\`;
- \`PATTERN_SELECTED\`.

Match or stratify at least:

- token count;
- domain;
- sequence length;
- gross difficulty/perplexity where practical.

Measure:

- target-state occupancy;
- transition rate;
- held-out probe transfer;
- stable-control capability;
- loss trajectory.

### 11.3 Closed-loop learning-progress controller

Only after open-loop data selection works.

At intervals:

\[
\theta_t
\rightarrow
\text{state probe}
\rightarrow
\text{mixture selection}
\rightarrow
\theta_{t+1}.
\]

Compare against:

- fixed random mixture;
- fixed "good" mixture;
- loss-progress-only controller;
- state-aware controller.

The main outcome is not final score alone. It is whether the controller increases desirable-state occupancy and reduces recovery cost without degrading matched conventional capability.

## 12. Hysteresis and path dependence

Construct two or more histories ending at matched token budgets and, where feasible, matched final data mixtures.

Examples:

\[
G \rightarrow P \rightarrow G
\]

versus

\[
P \rightarrow G.
\]

Then compare:

- behavioural margin;
- recovery cost;
- representation/operator state;
- optimizer state;
- downstream fine-tuning transfer.

A hysteresis claim requires history dependence after controlling the declared final conditions.

Independent-run variance alone is insufficient.

## 13. Optimizer programme

Do not start by retraining a large model with many optimizers.

First establish a controlled small-model regime in which competing shallow and transferable computations both occur.

Then compare, under matched model/data/compute:

1. AdamW baseline;
2. Muon-like update;
3. tempered Muon candidate;
4. manifold/normalized update where applicable;
5. MODULUS-derived update if an implementation with matched semantics exists.

Primary outcomes:

- time to first acquisition;
- desirable-state occupancy;
- transition frequency;
- recovery cost after a bad state;
- conventional loss;
- compute-normalized capability.

A useful optimizer wins only if the state-stability effect survives matching on conventional capability.

## 14. Architecture programme

Architecture comparisons are downstream of a validated controlled regime.

Candidate pairings:

- standard Transformer;
- nGPT / normalized hyperspherical state;
- RUNT where reversibility is relevant;
- later sparse/MoE variants if routing becomes a material state coordinate.

Questions:

- Does geometry constrain destructive state motion?
- Does reversibility lower recovery cost?
- Does architecture change occupancy or only shift when acquisition occurs?
- Are generalization-state transitions associated with different representation/operator signatures?

Match:

- active parameter count;
- training tokens;
- optimizer where possible;
- compute budget;
- context length;
- evaluation schedule.

Do not interpret lower loss variance as lower generalization-state variance without direct evidence.

## 15. Post-training transfer programme

Only after a pre-training state metric predicts held-out pre-training transitions.

Select at least:

- one strongly generalizing checkpoint;
- one strongly pattern-matching checkpoint;
- one terminal checkpoint.

Apply matched post-training.

Evaluate transfer to tasks not used for checkpoint selection.

Candidate outcomes:

- GPQA or equivalent reasoning transfer;
- robustness to shallow alignment/prefill attacks;
- out-of-context reasoning;
- other held-out GCL reasoning probes.

The central test is whether pre-training state predicts post-training generalization better than token count or ordinary pre-training capability.

## 16. Work-package decomposition

### GSD-WP00 — Source lock and evaluator fidelity

**Question:** Can the paper's probe semantics be reproduced exactly enough for a GCL checkpoint sweep?

**Deliverables:**
- exact source/version record;
- prompt templates;
- answer-span scoring implementation;
- dataset provenance;
- unit tests for span log-probability;
- small golden fixture.

**Exit gate:** independent replay yields identical scores on the golden fixture.

**Failure disposition:** \`EVAL_CONTRACT_UNRESOLVED\`.

### GSD-WP01 — 1B checkpoint harness

**Question:** Can many checkpoints be evaluated reproducibly and cheaply?

**Model:** OLMo 2 1B early-training first; OLMo 1B full-series second.

**Deliverables:**
- checkpoint enumerator;
- streaming evaluator;
- CSV/Parquet trace;
- W&B run;
- memory/runtime profile;
- exact revision manifest.

**Exit gate:** at least 20 checkpoints replay cleanly with deterministic score extraction.

**Failure disposition:** \`CHECKPOINT_PIPELINE_BLOCKED\`.

### GSD-WP02 — Transition catalogue

**Question:** Which probes show resolved non-monotone state movement?

**Deliverables:**
- soft-margin trajectories;
- uncertainty bands;
- state labels;
- candidate-transition table;
- stable-control trajectories;
- cross-probe correlation matrix.

**Exit gate:** every claimed transition satisfies the predeclared transition contract.

**Failure disposition:** either \`NO_TRANSITION_AT_THIS_SCALE\` or \`TRANSITIONS_FOUND\`.

### GSD-WP03 — Adversarial transition validation

**Mode:** independent blind from WP02 interpretation where possible.

**Question:** Are candidate transitions robust to prompt variants, sample resampling, answer-span implementation, and threshold choices?

**Deliverables:**
- alternate prompt/reordering tests;
- bootstrap replay;
- hard-versus-soft comparison;
- tokenizer/span audit;
- quantization sensitivity if quantized discovery was used.

**Exit gate:** surviving transitions promoted to \`CONFIRMED_TRANSITION\`.

### GSD-WP04 — CPS transition-local dynamics

**Question:** Does optimizer-state structure predict or explain confirmed transitions?

**Inputs:** exact transition windows with optimizer artifacts.

**Deliverables:**
- optimizer-state extraction;
- augmented-state probe implementation;
- transition-local traces;
- held-out predictive test.

**Exit gate:** report \`PREDICTIVE_SIGNAL\`, \`DESCRIPTIVE_ONLY\`, or \`NO_SIGNAL\`.

### GSD-WP05 — Mechanistic/operator localization

**Question:** Which internal structures move with a confirmed transition?

**Deliverables:**
- representation similarity;
- K-DIAGNOSTICS pack;
- head/layer/operator candidate signatures;
- pre-registered held-out replay.

**Exit gate:** candidate signature survives held-out transition testing or is rejected.

### GSD-WP06 — Recovery and Residual test

**Question:** Was the generalizing computation lost, inaccessible, or behaviourally suppressed?

**Deliverables:**
- R0–R4 recovery ladder;
- intervention costs;
- causal patching results;
- recovery matrix by checkpoint/probe;
- Residual-oriented interpretation.

**Exit gate:** transition classified at the strongest supported evidence level E0–E4.

### GSD-WP07 — Data-window attribution

**Question:** What training data moves the state?

**Deliverables:**
- source window identity where available;
- candidate positive/negative data classes;
- exact or approximate influence evidence;
- matched subsets for control experiments.

**Exit gate:** at least one attribution hypothesis survives a held-out micro-continuation test.

### GSD-WP08 — Open-loop and closed-loop curriculum control

**Question:** Can data choice change occupancy and recovery cost?

**Deliverables:**
- RANDOM / GENERALIZATION_SELECTED / PATTERN_SELECTED runs;
- multi-seed occupancy and transition statistics;
- stable-control metrics;
- closed-loop controller only after open-loop success.

**Exit gate:** controller effect survives seeds and matched conventional capability.

### GSD-WP09 — Optimizer-state stabilization

**Question:** Do update rules change transition dynamics?

**Deliverables:**
- matched AdamW baseline;
- Muon/tempered-Muon candidate;
- selected geometry/MODULUS candidate where implementable;
- compute-normalized state metrics.

**Entry gate:** WP08 or equivalent controlled regime established.

**Exit gate:** replicated state-stability difference after matching conventional capability.

### GSD-WP10 — Architecture stabilization

**Question:** Do nGPT/RUNT-style constraints change acquisition, occupancy, or recovery?

**Deliverables:**
- matched Transformer baseline;
- normalized/reversible candidate;
- state trajectory and recovery comparison.

**Entry gate:** a controlled small-model transition regime exists.

### GSD-WP11 — OLMo3 32B / post-training confirmation

**Question:** Does the GCL state description predict high-cost downstream generalization?

**Deliverables:**
- selected checkpoint trio;
- exact-paper-scale probe replay where feasible;
- matched post-training;
- held-out reasoning/alignment transfer.

**Entry gate:** at least one state metric has held-out predictive value at smaller scale.

**Failure disposition:** negative result is retained; do not search additional checkpoints post hoc without a new preregistration.

### GSD-WP12 — Synthesis and doctrine reconciliation

**Question:** What survived falsification?

**Deliverables:**
- evidence matrix by acquisition/persistence/accessibility/expression;
- accepted and rejected mechanism hypotheses;
- recommended cross-programme GCL consequences and explicit downstream handoffs;
- recommendation on candidate GCL-GS-00 admission/revision;
- paper-ready result only if evidence warrants it.

## 17. Gate structure

### Gate G0 — evaluator trustworthy

Requires:

- exact span scoring;
- prompt fixtures;
- reproducible checkpoint loading;
- stable controls.

Unlocks:
WP02.

### Gate G1 — transition real

Requires at least one \`CONFIRMED_TRANSITION\` based on soft margins.

Unlocks:
WP04, WP05, WP06.

If no small-model transition exists, the programme may escalate to OLMo3 7B without changing the claim that the small-model lane was negative.

### Gate G2 — state distinction informative

Requires at least one transition classified above E0, or a robust negative showing that candidate diagnostics fail.

Unlocks:
WP07 and targeted 7B/32B replication.

### Gate G3 — state controllable

Requires a matched data intervention that changes occupancy, transition rate, or recovery cost across seeds without merely destroying ordinary capability.

Unlocks:
WP09 and WP10.

### Gate G4 — stabilization replicates

Requires an optimizer or architecture effect that survives:

- matched compute;
- matched conventional capability;
- multiple seeds;
- held-out probes.

Unlocks:
WP11 scale-up.

### Gate G5 — downstream relevance

Requires state-informed checkpoint selection to predict held-out post-training transfer better than ordinary baselines.

Unlocks:
strong cross-programme claims and standards revision.

## 18. Independent-agent decomposition

The programme should exploit GCL's zero-context work-package mechanism after a confirmed transition packet exists.

Recommended blind lanes:

- **Lane A — behavioural auditor:** independently replay transition scoring.
- **Lane B — mechanistic localizer:** receives checkpoint pair and probe only; no preferred mechanism hypothesis.
- **Lane C — recovery attacker:** searches for the lowest-cost recovery intervention.
- **Lane D — falsifier:** attempts to explain the transition by prompt/tokenizer/metric artefact.
- **Lane E — optimizer analyst:** receives optimizer states without the other lanes' conclusions.

Synthesis is prohibited until required blind returns are durably received.

The protected transition packet should contain:

- checkpoint identities;
- exact prompts/datasets;
- environment lock;
- behavioural trace;
- allowed tools;
- claim firewall;
- return grammar.

## 19. Compute discipline

### Low-cost lane

Use for WP00–WP03:

- single T4/L4;
- 1B models;
- batched log-probability evaluation;
- no full activation dumps except targeted samples.

### Medium-cost lane

Use for WP04–WP08 and 7B confirmation:

- L4/A100;
- targeted layers/heads;
- CPU/NVMe activation or optimizer-state offload;
- small continuation budgets.

### High-cost lane

Use only after G4:

- 32B BF16 or equivalent;
- A100/H100 80 GB class or multi-GPU;
- matched post-training;
- strictly pre-registered checkpoint selection.

A failed gate is a compute-saving result.

## 20. Reproducibility contract

Every run must emit:

- model repository and exact revision/checkpoint;
- tokenizer revision;
- code commit;
- environment lock;
- random seeds;
- prompt/dataset digest;
- logits or sufficient score statistics;
- per-item soft margins;
- aggregate metrics;
- hardware;
- wall time;
- peak memory;
- W&B run identifier where used;
- CSV/Parquet outputs;
- compact PNG figures;
- claim-level disposition.

Large raw activations may be excluded from the durable repository if storage cost is excessive, but extraction code, indices, summary statistics, and checksums must remain replayable.

## 21. Statistical discipline

Minimum defaults:

- report per-item distributions, not only means;
- bootstrap intervals for soft margins;
- multiple seeds for training interventions;
- correct for repeated model/probe selection where inferential claims depend on it;
- separate exploratory discovery checkpoints from confirmatory hold-out checkpoints;
- do not select the "best layer" on the same transitions used to claim prediction;
- preserve negative probes and failed predictors.

The primary goal is structural evidence, not p-value accumulation.

## 22. Stop and pivot rules

### No transition at 1B

Disposition:
\`NO_TRANSITION_AT_THIS_SCALE\`.

Action:
move to OLMo3 7B with the validated evaluator. Do not reframe small-model noise as a positive result.

### No transition at 7B on paper probes

Action:
test whether the exact paper checkpoint region/probe set is accessible at 32B before abandoning replication.

### Transition disappears under soft margins

Disposition:
\`THRESHOLD_ARTEFACT\`.

Stop mechanistic analysis of that transition.

### Transition tracks ordinary capability collapse

Disposition:
\`GENERIC_DEGRADATION_NOT_GSD\`.

Do not promote it as generalization-state evidence.

### Quantization changes state labels

Use full-precision or documented higher-precision evaluation for claims.

### No predictor survives held-out testing

Disposition:
\`NO_PREDICTIVE_DIAGNOSTIC_YET\`.

Keep descriptive diagnostics; do not tune indefinitely on the same transition catalogue.

### Recovery requires substantial retraining

Do not claim persistence. Report the measured recovery curve and leave \`PERSISTENCE_UNRESOLVED\` unless stronger evidence exists.

## 23. Priority order

The execution priority is:

1. WP00 — evaluator fidelity.
2. WP01 — 1B harness.
3. WP02 — transition catalogue.
4. WP03 — adversarial validation.
5. WP06 — recovery/Residual test.
6. WP05 — mechanistic/operator localization.
7. WP04 — CPS where optimizer states permit.
8. WP07 — data attribution.
9. WP08 — curriculum control.
10. WP09/WP10 — optimizer and architecture stabilization.
11. WP11 — 32B/post-training confirmation.
12. WP12 — synthesis.

WP06 is intentionally ahead of broad mechanism fishing: if a bad checkpoint can be cheaply rescued, the programme immediately learns something about persistence/accessibility and avoids an undirected search over thousands of internal statistics.

## 24. Immediate first tranche

The first executable tranche should contain only WP00–WP03.

It should produce:

- a tested evaluator;
- an exact checkpoint manifest;
- at least one complete 1B trajectory;
- a transition catalogue;
- a falsification audit;
- a decision whether OLMo3 7B is warranted.

No optimizer or architecture comparison is authorized by the first tranche.

## 25. Success criteria for the programme

The programme is scientifically successful if it reaches any one of the following with strong evidence:

1. a reproducible transition with a causal accessibility/persistence explanation;
2. a robust negative result showing popular internal diagnostics fail to predict state changes;
3. a data/curriculum controller that changes state occupancy under matched capability;
4. an optimizer or architecture that reproducibly stabilizes transferable computation;
5. a state metric that predicts held-out post-training generalization;
6. evidence that the motivating phenomenon is narrower than initially believed.

The programme does not require confirming the motivating paper's preferred explanation.

## 26. Relationship to the GCL agenda

This programme operationalizes:

- **Residual:** recovery and minimal transferable structure;
- **CPS:** transition-local optimizer-state dynamics;
- **K-DIAGNOSTICS:** structured operator signatures;
- **Curriculum / ProgressSearch / MinCurr:** state-aware experience selection;
- **nGPT / RUNT:** geometry and reversibility as state-stability hypotheses;
- **Muon / MODULUS:** update geometry as a state-stability hypothesis;
- **Optionality:** recovery cost as a candidate training-time correction-capacity object.

The final unifying question is:

> **What determines which learned computation controls behaviour, what survives when control changes, and how can training preserve the transferable computations without sacrificing ordinary capability?**
