# GSD-WP05 — Mechanistic/operator localization receipt

Disposition: `WP05_SIGNATURE_VALIDATED_WITHIN_TRANSITION`

Programme-level exit status: `CROSS_TRANSITION_VALIDATION_NOT_YET_AVAILABLE`

Strongest programme-level evidence remains:
`E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

Hosted replay:
- experiment: `GSD-WP05-MECHANISTIC-OPERATOR-A100-001`
- run: `20261006T094253Z-3095`
- source commit: `1691e82ee6691e36bb3cfbeb4ffe619dff484805`
- hardware: NVIDIA A100-SXM4-40GB
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 3/3/3
- source payload SHA-256: `ccdb5c12c98ecde686791da9dc951c87b01d6399ea743f2469da0e54c839989e`
- job SHA-256: `5365d964be45b2fef6c2d9db85bc61281c095ff4a187d1e212ca7dc40660fde2`
- result SHA-256: `b72e4b3a6b394a7ef00535cbee2b062b9565598033c54fdf97e210cf80b57c96`
- protocol: `experiments/gsd-001/WP05_PROTOCOL.md`

## Preserved failed execution

The first frozen execution, run `20261006T093654Z-2786` from source
`9c3d840c78ad4003575ba71e453769b8a43c8392`, terminated
`RED_ENGINEERING` before producing a scientific disposition.

The failure was an implementation-only tensor-ordering defect: OLMo2 q/k normalization
was applied after reshaping the projected vector into attention heads. The native
normalizer expects the concatenated projection. The repair changed only normalization
ordering; the frozen partitions, selection rule, thresholds, checkpoints, and claim
boundary were unchanged. Fresh exact-head replay followed.

## Data and claim boundary

All 287 `truthy_answer/surprising_truth` test rows were already consumed by the
programme before WP05.

WP05 therefore made no programme-fresh holdout claim.

Within WP05:
- rows 0–63 were used only for signature discovery;
- rows 64–127 were frozen as the evaluation partition before execution;
- the evaluation partition was not used to select the candidate layer/head.

Because GSD currently has only one `CONFIRMED_TRANSITION`, this is a
within-transition validation. It is not a cross-transition predictive test.

## Behavioural sanity gate

The previously confirmed transition reproduced independently on both WP05 partitions.

2k source:
- discovery: GENERALIZING, mean margin +1.1758, CI95 [+1.0674,+1.2852]
- evaluation: GENERALIZING, mean margin +1.1699, CI95 [+1.0927,+1.2471]

3k flank:
- discovery: PATTERN_MATCHING, mean margin -2.0566, CI95 [-2.1748,-1.9336]
- evaluation: PATTERN_MATCHING, mean margin -2.1699, CI95 [-2.2647,-2.0732]

4k target:
- discovery: PATTERN_MATCHING, mean margin -2.1426, CI95 [-2.2715,-2.0088]
- evaluation: PATTERN_MATCHING, mean margin -2.3486, CI95 [-2.4483,-2.2461]

Exact continuation-boundary gate: pass.

## Frozen representation signature

Discovery selected transformer layer 13.

Discovery:
- cosine drift 2k→3k: 0.27905
- cosine drift 3k→4k: 0.16652
- transition-locality difference: +0.11253

Evaluation at the frozen same layer:
- cosine drift 2k→3k: 0.28153
- cosine drift 3k→4k: 0.16692
- transition-locality difference: +0.11461
- evaluation percentile among layers: 1.000
- discovery/evaluation 2k→3k change-vector cosine: 0.99550

All preregistered representation criteria passed.

## Frozen K-DIAGNOSTICS operator signature

Discovery selected:
- operator: query projection
- layer: 14
- head: 2

Discovery:
- cosine drift 2k→3k: 0.62967
- cosine drift 3k→4k: 0.29064
- transition-locality difference: +0.33902

Evaluation at the frozen same operator:
- cosine drift 2k→3k: 0.58010
- cosine drift 3k→4k: 0.28491
- transition-locality difference: +0.29519
- evaluation percentile among all Q/K candidates: 0.99609
- discovery/evaluation 2k→3k change-vector cosine: 0.99752

All preregistered operator criteria passed.

## Adjudication

The correct bounded disposition is:

`WP05_SIGNATURE_VALIDATED_WITHIN_TRANSITION`

The result supports a reproducible representation/operator change aligned with the
confirmed behavioural transition. In particular, layer 13 residual representation and
layer-14 query head 2 are now frozen candidate signatures for future independent
transition testing.

It does **not** establish:
- that either signature is causal;
- literal circuit identity;
- mechanism persistence;
- mechanism replacement;
- cross-transition prediction.

WP06 already supports E2 accessibility/control shift, so WP05 does not raise the
programme evidence ceiling beyond E2.

## Evidentiary boundary

The programme-level WP05 exit gate calls for held-out **transition** validation.
No second confirmed transition currently exists. The other catalogue candidates remain
unconfirmed and may not be promoted merely to satisfy this gate.

Therefore the remaining WP05 obligation is:

`VALIDATE_FROZEN_LAYER13_Q14H2_SIGNATURE_ON_NEXT_INDEPENDENT_CONFIRMED_TRANSITION`

Until such a transition exists:

`NO_CROSS_TRANSITION_PREDICTIVE_CLAIM`.
