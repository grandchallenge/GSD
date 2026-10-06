# GSD-WP05U U1 — directional causal intervention receipt

Disposition: `NO_HEAD2_CAUSAL_EFFECT`

Programme evidence ceiling remains:
`E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

Hosted execution:
- experiment: `GSD-WP05U-U1-DIRECTIONAL-CAUSAL-A100-001`
- run: `20261006T111223Z-3546`
- source commit: `ed980a16a6d59f99195a7a1678677b4e2690633a`
- hardware: NVIDIA A100-SXM4-40GB
- engineering status: `GREEN_ENGINEERING`
- source payload SHA-256: `2eaae64e4ef6ae4756180f1765c6fd4f78b376fd5d3647b699a86031b549df33`
- hosted job SHA-256: `a25593f2ed2fc733a77aaeff5c9279ee87ecc1ba8734cafd22745ce29dd4b3e5`
- host result SHA-256: `3f92d54cdc74fb7382512809c7085b735647c493813741b911cfdaa45e16c110`
- scientific result SHA-256: `4f654eb85f3b17e42b79c0c02698168a6f9d06965e9cf4e89044e6ae7fe869e2`
- exact-boundary gate: pass

## Frozen test

The inherited WP05 marker was Q / layer 14 / head 2.

Discovery rows 0–63 defined the native post-q-normalization 2k→3k transition direction.
Evaluation rows 64–127 were scored at the 4k checkpoint.

Interventions:
- target reverse: add `-1.0 * delta` to Q14/H2 at the final prompt token;
- target amplify: add `+1.0 * delta`;
- structural reverse controls: Q14/H1, Q14/H3, Q13/H2, Q15/H2.

All rows were already programme-consumed before this tranche. This is not a programme-fresh holdout and does not create a second transition.

## Result

4k baseline:
- mean behavioural margin: -2.34863
- state: PATTERN_MATCHING

Q14/H2 reverse:
- mean margin: -2.34375
- paired mean delta: +0.00488
- paired bootstrap CI95: [-0.00488, +0.01465]
- state: PATTERN_MATCHING

Q14/H2 amplify:
- mean margin: -2.35352
- paired mean delta: -0.00488
- paired bootstrap CI95: [-0.01465, +0.00488]
- state: PATTERN_MATCHING

Structural reverse controls:
- Q13/H2: -0.01465, CI95 [-0.02734,-0.00293]
- Q14/H1: -0.02051, CI95 [-0.02930,-0.01074]
- Q14/H3: +0.05957, CI95 [+0.04785,+0.07129]
- Q15/H2: -0.01758, CI95 [-0.02637,-0.00879]

The preregistered directional-causal threshold required target reverse >= +0.25, target amplify <= -0.25, and target reverse to exceed every structural-control reverse effect by >= 0.10. None of these conditions was met.

## Adjudication

The exact result is:

`NO_HEAD2_CAUSAL_EFFECT`

under the declared final-prompt-token post-q-normalization intervention surface.

This falsifies the simple interpretation that the unusually transition-local Q14/H2 mean direction is itself a strong causal control coordinate for the observed behavioural state.

It does not establish that Q14/H2 is irrelevant. The original WP05 localization remains valid as a reproducible within-transition marker. It may reflect:
- a downstream/readout correlate;
- a distributed late-query subspace;
- a change whose causal effect requires coordinated heads/layers/tokens;
- a marker of a larger circuit rather than a single-head lever.

The Q14/H3 control effect is not promoted to a causal discovery: it was a structural control, lacked a preregistered directional pair, and was not selected under an independent causal-discovery protocol.

## Utility consequence

Do not promote `Q14H2_WITHIN_TRANSITION_MARKER` to `Q14H2_DIRECTIONAL_CAUSAL_SURFACE`.

Broaden utility by testing:
1. independent-transition and scale transfer using normalized depth and late-query subspaces rather than literal head identity;
2. temporal precursor value at denser checkpoints;
3. WP07 data-attribution coupling between training data, internal subspace motion, and behavioural state.

The original programme-level WP05 obligation remains:

`VALIDATE_FROZEN_LAYER13_Q14H2_SIGNATURE_ON_NEXT_INDEPENDENT_CONFIRMED_TRANSITION`.

No E3/E4 promotion is authorized by U1.
