# GSD-WP05 — Mechanistic/operator localization protocol

Status: frozen before execution.

Parent: `GSD-WP05 — Mechanistic/operator localization`.

Confirmed behavioural target:
- task: `truthy_answer/surprising_truth`;
- source: `stage1-step2000-tokens5B` = GENERALIZING;
- localization flank: `stage1-step3000-tokens7B` = PATTERN_MATCHING;
- target: `stage1-step4000-tokens9B` = PATTERN_MATCHING;
- exact-boundary transition confirmation is inherited from WP03R;
- strongest programme evidence before WP05 is E2 `ACCESSIBILITY_SHIFT_SUPPORTED`.

## Claim boundary

WP05 asks which internal structures move with the confirmed transition. It does not reopen WP06 persistence adjudication.

All 287 surprising_truth test rows have already been consumed by the programme. WP05 therefore makes no programme-fresh-holdout claim.

Rows 0-63 are the WP05 discovery partition. Rows 64-127 are frozen before execution as the WP05 evaluation partition and are held out only from WP05 signature selection.

Because the programme currently has only one confirmed transition, a successful WP05 result is within-transition validation, not evidence that the signature predicts an independent transition.

## Probe surface

Use the canonical seed-0 `base_k8` misleading-demonstration context.

For each item, internal statistics are measured at the final prompt token, before either candidate answer is appended. This prevents answer-token contamination of the mechanistic signature.

Checkpoints:
- 2k source;
- 3k transition flank;
- 4k post-transition target.

Precision: BF16 on a BF16-capable accelerator.

## Representation diagnostics

For every transformer layer input, compute the mean final-prompt-token hidden vector separately on discovery and evaluation partitions.

For each layer and partition, record:
- cosine drift 2k -> 3k;
- cosine drift 3k -> 4k;
- transition-locality difference = drift(2k,3k) - drift(3k,4k).

Select exactly one representation layer on discovery: the layer with the largest transition-locality difference.

No reselection is allowed after evaluation.

Evaluation success for the selected representation layer requires:
1. drift(2k,3k) >= 1.25 * drift(3k,4k);
2. its evaluation transition-locality difference is in the top quartile of all layers;
3. cosine between the discovery and evaluation 2k->3k change vectors is >= 0.5.

## K-DIAGNOSTICS operator surface

At every layer, project the final-prompt-token layer input through the checkpoint's native attention q_proj and k_proj, including native q_norm/k_norm when present.

Treat every query head and key/value head as a separate operator candidate.

For each candidate and partition, record:
- cosine drift 2k -> 3k;
- cosine drift 3k -> 4k;
- transition-locality difference.

Select exactly one operator candidate on discovery from the union of Q and K heads: the candidate with largest transition-locality difference among candidates whose 2k->3k discovery drift is at least the median discovery drift.

Evaluation success for the frozen operator candidate requires:
1. drift(2k,3k) >= 1.25 * drift(3k,4k);
2. its evaluation transition-locality difference is in the top decile of all Q/K candidates;
3. cosine between discovery and evaluation 2k->3k change vectors is >= 0.5.

## Behavioural sanity gate

On both discovery and evaluation partitions:
- 2k must classify GENERALIZING;
- 3k must classify PATTERN_MATCHING;
- 4k must classify PATTERN_MATCHING;
- answer-span boundaries must remain exact.

If this gate fails, disposition is `WP05_BASELINE_UNRESOLVED`.

## Dispositions

`WP05_SIGNATURE_VALIDATED_WITHIN_TRANSITION`
- behavioural sanity gate passes;
- representation candidate passes all frozen evaluation criteria;
- operator candidate passes all frozen evaluation criteria.

This supports a reproducible E1 representation/operator change aligned with the already-confirmed transition, but does not establish cross-transition prediction or causal mechanism identity.

`WP05_SIGNATURE_PARTIAL`
- behavioural sanity gate passes;
- exactly one of representation/operator candidates passes.

`WP05_SIGNATURE_REJECTED`
- behavioural sanity gate passes;
- neither candidate passes.

`WP05_BASELINE_UNRESOLVED`
- behavioural sanity gate fails.

`EXACT_BOUNDARY_CONTRACT_FAILED`
- answer-span boundary audit fails.

## Programme interpretation

WP05 cannot raise the programme above the existing E2 evidence ceiling by itself.

A validated within-transition signature may nominate a fixed diagnostic for future independent-transition testing. Until another confirmed transition exists, `NO_CROSS_TRANSITION_PREDICTIVE_CLAIM` remains mandatory.
