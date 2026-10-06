# GSD-WP05U U2A — OLMo3 7B coarse transfer scan

Status: frozen before execution.

Purpose: seek an independent 7B behavioural transition under the validated GSD evaluator before testing whether the WP05 late-query signature transfers across scale.

Model:
- repository: `allenai/Olmo-3-1025-7B`
- architecture: OLMo3 7B
- Stage-1 checkpoint surface is used only.

Task family:
- `truthy_answer`
- primary target remains `truthy_answer/surprising_truth`.

Frozen coarse checkpoints:
- `stage1-step0`
- `stage1-step282000`
- `stage1-step565000`
- `stage1-step705000`
- `stage1-step846000`
- `stage1-step1130000`
- `stage1-step1413814`

These were chosen before execution to span the released Stage-1 trajectory. No checkpoint may be added or removed after seeing behavioural results without a new preregistration.

Evaluation:
- one canonical seed;
- max 64 evaluation items per task for coarse discovery;
- exact model and dataset SHAs retained;
- soft margins and task-level state labels are primary.

## Claim firewall

U2A is discovery only.

A coarse reversal is not a `CONFIRMED_TRANSITION`.
Any candidate must pass the existing WP03 exact-boundary, stable-control, prompt-variant and resampling contract before it can:
- satisfy the WP05 cross-transition exit gate;
- support a cross-scale signature claim;
- trigger mechanistic/subspace promotion.

No literal OLMo2 head-number correspondence is assumed. OLMo2 Q14/H2 is treated as a marker coordinate, not an architecture invariant.

## Decision rule

If no resolved state reversal appears:
- disposition `NO_7B_COARSE_TRANSITION`;
- retain the negative result;
- do not post-hoc search arbitrary checkpoints.

If a resolved state reversal appears:
- disposition `7B_CANDIDATE_TRANSITION`;
- freeze the smallest bracketing interval;
- launch a finer behavioural confirmation tranche inside that interval;
- only after confirmation test normalized depth and late-query subspace transfer.

The target transfer object is a functional late-query subspace, not literal "head 2".
