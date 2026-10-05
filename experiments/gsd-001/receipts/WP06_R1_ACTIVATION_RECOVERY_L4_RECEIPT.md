# GSD-WP06-R1 — Activation-space recovery receipt

Disposition: `R1_DISCOVERY_ONLY_REJECTED`

Strongest supported evidence level remains:
`E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

Hosted run:
- experiment: `GSD-WP06-R1-ACTIVATION-RECOVERY-L4-001`
- run: `20261005T113832Z-997`
- source commit: `8cdc256989fc036ba657ae623ee856284d5e468b`
- hardware: NVIDIA L4
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 3/3/3
- payload SHA-256: `d264ab8ff33301a95a0d9ec51cc9c3457d92e098cccfe3e876db308dcd99d806`
- job SHA-256: `6cfce743a3ec1d16f97917268126b39a50e7e735db237b37c8777e38aa9ccab6`
- result SHA-256: `949501a27fd5a9442a9b43be4f5eb79a8ece012b73c83a6a683571deb27fa994`
- protocol: `experiments/gsd-001/WP06_R1_PROTOCOL.md`

Confirmed behavioural transition under test:
- task: `truthy_answer/surprising_truth`
- source step 2000: GENERALIZING
- flank step 3000: PATTERN_MATCHING
- target step 4000: PATTERN_MATCHING
- original hostile context: seed-0 base_k8 demonstrations
- broad stable controls: pass
- exact answer boundary: pass

## Discovery split — rows 0-63

Unpatched:
- source 2k: GENERALIZING, mean margin +1.174, CI95 [+1.066, +1.282]
- target 4k: PATTERN_MATCHING, mean margin -2.146, CI95 [-2.276, -2.014]

Source-2k residual patch into target-4k:
- after layer 4: PATTERN_MATCHING, mean -2.157
- after layer 8: PATTERN_MATCHING, mean -1.337
- after layer 12: GENERALIZING, mean +0.219, CI95 [+0.083, +0.361]
- after layer 16: GENERALIZING, mean +0.555, CI95 [+0.434, +0.675]

The preregistered rule selects the earliest recovering point: layer 12.

Intervention cost at the selected point:
- source layers imported: 12
- residual vectors patched per sequence: 1.0 mean
- hidden dimensions per vector: 2048

## Held-out confirmation — rows 64-127

Unpatched:
- source 2k: GENERALIZING, mean +1.163, CI95 [+1.087, +1.240]
- target 4k: PATTERN_MATCHING, mean -2.351, CI95 [-2.449, -2.248]
- flank 3k: PATTERN_MATCHING, mean -2.167, CI95 [-2.261, -2.068]

Matched layer-12 source->target patch:
- target 4k: UNCERTAIN, mean +0.127, CI95 [-0.0010, +0.254]
- flank 3k: PATTERN_MATCHING, mean -0.391, CI95 [-0.501, -0.275]

Thus the selected intervention does not satisfy the held-out GENERALIZING gate.

Controls:
- target->target identity patch: PATTERN_MATCHING
- maximum per-item identity margin drift: 0.0
- item-shuffled source patch at 4k: UNCERTAIN, mean +0.084, CI95 [-0.053, +0.227]

The implementation control is exact. The matched patch strongly shifts the target margin but does not meet the frozen recovery criterion, and the shuffled patch also produces a sizeable non-specific shift.

## Interpretation

A late residual-stream state from the 2k generalizing checkpoint contains information capable of moving the 4k model substantially toward generalizing behaviour. This effect is strong in discovery data but does not confirm on the held-out split under the preregistered layer-selection rule.

Therefore:
- R1 activation recovery is **not confirmed**;
- E3 mechanism-persistence evidence is **not authorized**;
- the strongest programme-level result remains E2 accessibility/expression shift from R0.

The layer-16 discovery result must not be tested post hoc on the already-used held-out split. Any direct layer-16 confirmation would require a separately frozen fresh-data replay.

Next action should not reinterpret this near miss as success. Proceed to the next predeclared recovery/localization boundary with the negative R1 result preserved.
