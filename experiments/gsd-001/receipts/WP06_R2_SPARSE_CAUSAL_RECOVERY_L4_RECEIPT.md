# GSD-WP06-R2 — Sparse causal recovery receipt

Disposition: `NO_R2_RECOVERY`

Strongest supported evidence level remains:
`E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

Hosted run:
- experiment: `GSD-WP06-R2-SPARSE-CAUSAL-RECOVERY-L4-001`
- run: `20261006T070241Z-1133`
- source commit: `8724c4e0cd01404cb724c9ce9fbafe9016da921a`
- hardware: NVIDIA L4
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 3/3/3
- payload SHA-256: `03c17eba47d31f8a9b42785f9dbb6e0ccee038905994e0df5501c6fef9c74860`
- job SHA-256: `46fa60cafcec3b8cf428fe5ad9f4dd34653438abd85652b07de297bf03de175e`
- result SHA-256: `917e9eb5ab8bb8e6fe18016a9dc1f95e1804decfe760f53d81c01d46d996f7c0`
- protocol: `experiments/gsd-001/WP06_R2_PROTOCOL.md`

## Fresh-data discipline

The dataset contains 287 test rows.

Previously consumed before R2:
- rows 0-127.

R2 used only fresh rows:
- discovery: 128-191;
- held-out baseline confirmation: 192-255.

Still untouched and reserved:
- rows 256-286.

## Baseline on fresh rows

Discovery rows 128-191:
- source step 2k: GENERALIZING, mean margin +1.134, CI95 [+1.028,+1.244]
- target step 4k: PATTERN_MATCHING, mean margin -2.446, CI95 [-2.542,-2.357]

Held-out rows 192-255:
- source step 2k: GENERALIZING, mean margin +1.263, CI95 [+1.160,+1.366]
- target step 4k: PATTERN_MATCHING, mean margin -2.245, CI95 [-2.348,-2.140]
- flank step 3k: PATTERN_MATCHING, mean margin -2.141, CI95 [-2.231,-2.051]

Thus the confirmed behavioural state distinction reproduces on the fresh R2 tranche.

## Sparse module search

Exactly one post-normalized module contribution was patched at a time at answer-prediction positions.

Frozen search order:
- layer12_attention
- layer12_mlp
- layer13_attention
- layer13_mlp
- layer14_attention
- layer14_mlp
- layer15_attention
- layer15_mlp
- layer16_attention
- layer16_mlp

Discovery dispositions:
- layer12_attention: PATTERN_MATCHING, mean -1.522
- layer12_mlp: PATTERN_MATCHING, mean -2.237
- layer13_attention: PATTERN_MATCHING, mean -2.404
- layer13_mlp: PATTERN_MATCHING, mean -2.235
- layer14_attention: PATTERN_MATCHING, mean -2.630
- layer14_mlp: PATTERN_MATCHING, mean -2.186
- layer15_attention: PATTERN_MATCHING, mean -1.900
- layer15_mlp: PATTERN_MATCHING, mean -2.587
- layer16_attention: PATTERN_MATCHING, mean -2.275
- layer16_mlp: PATTERN_MATCHING, mean -2.134

No candidate produced GENERALIZING behaviour on the fresh discovery set.

The strongest sparse shift was layer-12 attention:
- target baseline mean: -2.446
- patched mean: -1.522

This is a material directional shift but remains confidently PATTERN_MATCHING.

Because no discovery candidate qualified, the preregistered protocol correctly performed no selected-module held-out causal confirmation and no post-selection identity/shuffle test.

## Interpretation

R2 rejects the hypothesis that the recoverable 2k generalizing state can be restored by replacing one late post-normalized attention or MLP contribution from layers 12-16 under the declared sparse module class.

This narrows the persistence hypothesis but does not establish E4 mechanism change.

In particular:
- R0 still shows behavioural accessibility under zero-shot factual instruction;
- R1 shows a broad late residual intervention can move the target strongly toward generalizing but fails held-out confirmation;
- R2 shows that this effect does not reduce to any one tested late attention/MLP contribution.

Strongest programme-level evidence therefore remains E2 accessibility/expression shift.

Next recovery tier:
`R3 — parameter-light recovery`.

Rows 256-286 remain untouched and may be used only under a new frozen preregistration.
