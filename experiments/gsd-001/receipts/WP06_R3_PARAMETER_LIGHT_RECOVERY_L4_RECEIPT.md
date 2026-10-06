# GSD-WP06-R3 — Parameter-light recovery receipt

Disposition: `R3_BASELINE_UNRESOLVED`

Primary reserve signal: strong, specific, but not promotable under the frozen R3 control gate.

Hosted run:
- experiment: `GSD-WP06-R3-PARAMETER-LIGHT-RECOVERY-L4-001`
- run: `20261006T074410Z-1718`
- source commit: `d0a06aebd73fb7b28b9015d4ae0ad50ac7e5e110`
- hardware: NVIDIA L4
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 2/2/2
- payload SHA-256: `f121c2d9019eec29a7df87e1433a1305490f434f69662fc1a341e4a602842302`
- job SHA-256: `b45bd93585ff7fbdf0b6198945d54180369d5a7af107ad5c27c76a4aec9c91d0`
- result SHA-256: `a0c883c92f7b15543214db16c17b43ada2caef1d05cde39e069f82a8113055ce`
- protocol: `experiments/gsd-001/WP06_R3_PROTOCOL.md`

## Frozen intervention

For each of seeds 0, 1, 2:
- one rank-1 LoRA adapter;
- target module: layer-12 `self_attn.o_proj`;
- 4096 trainable parameters;
- trainable fraction: 2.7584e-6 of model parameters (~0.000276%);
- one 64-example epoch;
- 32 two-item optimizer steps;
- AdamW, lr 3e-3;
- no other model parameter changed.

Matched reversed-label adapters used the identical architecture, data order, and update budget.

## Untouched reserve baseline — rows 256-286

These 31 Truthy rows had not been used by R0-R2.

Source 2k:
- GENERALIZING
- mean margin +1.284
- CI95 [+1.107,+1.468]
- positive fraction 1.0

Unadapted target 4k:
- PATTERN_MATCHING
- mean margin -2.290
- CI95 [-2.442,-2.135]
- positive fraction 0.0

The behavioural transition therefore reproduces on the final untouched reserve.

## Factual-treatment adapters

Seed 0:
- GENERALIZING
- mean margin +8.981
- CI95 [+8.814,+9.155]

Seed 1:
- GENERALIZING
- mean margin +9.116
- CI95 [+8.949,+9.287]

Seed 2:
- GENERALIZING
- mean margin +8.929
- CI95 [+8.762,+9.101]

Thus 3/3 parameter-light treatment adapters restore the generalizing answer on the untouched reserve.

## Reversed-label matched controls

Seed 0:
- PATTERN_MATCHING
- mean margin -10.892

Seed 1:
- PATTERN_MATCHING
- mean margin -10.936

Seed 2:
- PATTERN_MATCHING
- mean margin -10.743

Thus 0/3 reversed-label controls recover generalizing behaviour; they push strongly in the intended opposite direction.

This establishes that the small update is highly task-directional rather than a generic optimization disturbance.

## Ordinary-capability control gate

Frozen R3 required both conventional controls to be confidently GENERALIZING before and after treatment.

Unadapted target 4k:
- sentiment: GENERALIZING, mean +0.219, CI95 [+0.059,+0.375]
- arithmetic: UNCERTAIN, mean -0.386, CI95 [-1.020,+0.272]

Treatment adapters:
- sentiment remains GENERALIZING for all 3 seeds;
- arithmetic remains UNCERTAIN for all 3 seeds.

The arithmetic control therefore fails at the **baseline**, before adaptation.

Because the frozen R3 success rule required a confidently GENERALIZING base arithmetic control, the scientific disposition is correctly fail-closed as:

`R3_BASELINE_UNRESOLVED`

This is not evidence that the adapter caused generic damage: the arithmetic control was already unresolved before adaptation, and its post-adapter means are actually slightly less negative. But the frozen gate cannot be retroactively relaxed.

## Interpretation

The primary result is a strong parameter-light recovery signal:
- only 4096 trainable parameters;
- one short pass over 64 already-consumed items;
- 3/3 independent treatment seeds recover;
- 0/3 direction-reversed controls recover;
- final untouched reserve moves from strongly PATTERN_MATCHING to strongly GENERALIZING.

However R3 is intrinsically ambiguous between:
- recovery of a pre-existing latent computation; and
- rapid task-specific relearning through a tiny adapter.

The frozen control gate is unresolved, so no E3 persistence promotion is authorized.

Strongest programme-level evidence remains:
`E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

## Next action

Run a narrow **R3 control-gate repair** only:
- do not retrain or reinterpret the Truthy reserve result;
- do not reopen rows 256-286 for tuning;
- use the previously established 128-item conventional stable-control suite;
- reproduce the same fixed adapter-training recipe and test whether ordinary sentiment/arithmetic capability remains stable at adequate control-sample size;
- preserve the original R3 disposition as immutable.

No R4 escalation should occur until that control-gate repair is resolved.
