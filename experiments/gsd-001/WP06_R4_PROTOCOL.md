# GSD-WP06-R4 — Continued-training recovery

Status: frozen before execution.

Parent work package: `GSD-WP06 — Recovery and Residual test`.

Predecessor boundary:
- R0 supports E2 accessibility/expression shift.
- R1 did not confirm activation-space recovery on held-out data.
- R2 found no recovery under the declared one-module sparse class.
- R3/R3C established robust parameter-light recovery with persistence unresolved.
- No E3 mechanism-persistence promotion is currently authorized.

R4 purpose:
measure the cost curve for full-model selected-data continuation from the bad
checkpoint. R4 is a control/recovery measurement and is the weakest recovery
tier for persistence claims.

## Data boundary

All 287 `truthy_answer/surprising_truth` test rows have already been consumed
by the programme.

R4 therefore makes **no fresh-holdout claim**.

R4 uses:
- continuation-training rows: 128–191;
- recovery-curve evaluation rows: 192–255.

Rows 192–255 are separated from R4 training but were previously used by the
programme. They must be described only as the R4 evaluation set, not as fresh
or held-out evidence.

## Starting checkpoint

- target: `stage1-step4000-tokens9B`;
- source reference only: `stage1-step2000-tokens5B`.

Required pre-training-state check on the R4 evaluation set:
- source 2k = GENERALIZING;
- target 4k = PATTERN_MATCHING.

## Continued-training intervention

All target-model parameters are trainable.

Optimizer:
- AdamW;
- learning rate: 1e-5;
- weight decay: 0;
- gradient clip norm: 1.0;
- BF16 model/gradients;
- no scheduler;
- no optimizer-state reuse from original pre-training.

Batching:
- 2 items per optimizer step;
- one pass over the 64 continuation items;
- exactly 32 optimizer steps maximum.

Training loss:
- causal next-token negative log-likelihood;
- loss is applied only to the answer-continuation token positions;
- prompt and demonstration tokens receive no direct loss;
- original seed-0 `base_k8` misleading-demonstration context is retained.

This is targeted selected-data continuation, not reconstruction of the original
pre-training stream.

## Matched continuation lanes

Seeds: 0, 1, 2.

For each seed, use the same shuffled item order in both lanes.

### GENERALIZATION_SELECTED
Train on the predefined correct/factual answer continuation.

### PATTERN_SELECTED
Train on the predefined incorrect/pattern answer continuation.

There are therefore six full-model continuations total.

## Recovery curve

Evaluate the R4 evaluation set at optimizer steps:

`0, 1, 2, 4, 8, 16, 32`.

For every lane/seed record:
- soft-margin state;
- mean margin;
- bootstrap CI;
- exact continuation-boundary status;
- cumulative answer tokens processed.

For a GENERALIZATION_SELECTED seed, the recovery step is the earliest frozen
step at which the R4 evaluation set is GENERALIZING.

No interpolation between frozen step checkpoints is allowed.

## Conventional-capability gate

Before continuation, score the target checkpoint on the same 128-item
conventional suite used by R3C:
- ordinary SST-2 sentiment;
- conventional arithmetic;
- held-out review language-model token log-likelihood.

At the first recovery step of each GENERALIZATION_SELECTED seed, score the same
128 examples and compare them pairwise against the unadapted target.

A family is a resolved degradation iff the upper endpoint of its paired
adapter-minus-base 95% bootstrap CI is below zero.

A recovery seed has `control_ok` only if zero of the three conventional
families shows resolved degradation.

## Predeclared cost

For a recovered seed:

`C(delta) = (trainable_parameters, optimizer_steps, answer_tokens_processed)`.

R4 recovery cost is reported as the per-seed tuple at the earliest frozen
recovery step. No scalar weighting of the tuple is introduced post hoc.

## Dispositions

- `R4_CONTINUED_TRAINING_RECOVERY`
  - source/target baseline resolves;
  - at least 2/3 GENERALIZATION_SELECTED seeds recover by step 32;
  - every recovered treatment seed has `control_ok: true`;
  - 0/3 PATTERN_SELECTED seeds become GENERALIZING;
  - exact answer boundaries pass.

- `R4_GENERIC_DAMAGE`
  - recovery occurs but any recovered treatment seed shows resolved
    conventional-capability degradation.

- `R4_NONSPECIFIC_UPDATE`
  - any PATTERN_SELECTED seed becomes GENERALIZING.

- `R4_PARTIAL_RECOVERY`
  - exactly 1/3 factual seeds recovers and controls pass.

- `NO_R4_RECOVERY`
  - 0/3 factual seeds recover by step 32.

- `R4_BASELINE_UNRESOLVED`
  - source/target baseline state distinction fails on the R4 evaluation set.

- `EXACT_BOUNDARY_CONTRACT_FAILED`

## Interpretation

A positive R4 result measures continued-training recovery cost. It does not
establish persistence of the original computation because full-model training
can relearn or reconstruct the behaviour.

No E3 promotion follows from R4 alone.

A negative R4 result also does not establish E4 mechanism change; it only
bounds recovery under this declared continuation recipe and 32-step budget.

After R4, WP06 should be classified at the strongest evidence level already
supported by the entire R0–R4 ladder, without retroactively relaxing any tier.
