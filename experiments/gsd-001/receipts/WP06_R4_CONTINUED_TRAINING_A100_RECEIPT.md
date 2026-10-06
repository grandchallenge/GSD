# GSD-WP06-R4 — Continued-training recovery receipt

Disposition: `R4_GENERIC_DAMAGE`

Strongest programme-level evidence remains:
`E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

Hosted run:
- experiment: `GSD-WP06-R4-CONTINUED-TRAINING-A100-001`
- run: `20261006T082807Z-2169`
- source commit: `ae4c38e59363ebb552463759c16c829be5826aea`
- hardware: NVIDIA A100-SXM4-40GB
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 2/2/2
- payload SHA-256: `94ff5bd1b90be5ab8c687f698dbadf36cabebef986abd607d2ba2dbbd9183dd3`
- job SHA-256: `c9ccdbdaba0e3e50ad98cb5dc88a790d59f665852d43fa03f049e111c22da935`
- result SHA-256: `2d2e91e2b11b84e2883665a7f15d359f9667948563c86fd215a9e81f71a54057`
- protocol: `experiments/gsd-001/WP06_R4_PROTOCOL.md`

## Data boundary

All 287 `truthy_answer/surprising_truth` test rows were already consumed before R4.

R4 therefore made no fresh-holdout claim.

R4 used:
- continuation training: rows 128-191;
- recovery-curve evaluation: rows 192-255.

The evaluation rows were separated from R4 training but were previously used by the programme.

## Baseline state on the R4 evaluation set

Source step 2k:
- GENERALIZING
- mean margin +1.2783
- CI95 [+1.1777,+1.3809]

Target step 4k:
- PATTERN_MATCHING
- mean margin -2.2480
- CI95 [-2.3486,-2.1406]

Exact continuation-token boundary gate: pass.

## GENERALIZATION_SELECTED continuation

All target-model parameters were trainable:
- trainable parameters: 1,484,916,736
- AdamW learning rate: 1e-5
- batch size: 2 items
- maximum budget: 32 optimizer steps
- loss: answer-token causal NLL

All three factual lanes recovered at the same frozen checkpoint:

Seed 0:
- step 0: PATTERN_MATCHING, mean -2.2480
- step 1: PATTERN_MATCHING, mean -0.1299
- step 2: GENERALIZING, mean +1.8398

Seed 1:
- step 0: PATTERN_MATCHING, mean -2.2480
- step 1: PATTERN_MATCHING, mean -0.1221
- step 2: GENERALIZING, mean +1.8232

Seed 2:
- step 0: PATTERN_MATCHING, mean -2.2480
- step 1: UNCERTAIN, mean -0.1123
- step 2: GENERALIZING, mean +1.8994

All three remain strongly GENERALIZING through step 32.

The frozen R4 recovery-cost tuple is therefore identical for all three treatment seeds:

`C(delta) = (1,484,916,736 trainable parameters, 2 optimizer steps, 4 answer tokens processed)`.

This is an extremely short continued-training recovery curve, but it uses the full parameter set and therefore does not imply persistence.

## PATTERN_SELECTED matched controls

The three matched opposite-direction continuations never become GENERALIZING.

At step 2 their mean margins are approximately:
- seed 0: -6.018
- seed 1: -6.222
- seed 2: -6.146

By step 32 they are approximately:
- seed 0: -11.794
- seed 1: -12.019
- seed 2: -12.129

Thus the continuation direction is highly causal and specific: factual continuation restores the factual/generalizing state while pattern-selected continuation drives the same model further into the pattern state.

## Conventional-capability gate at the first recovery point

The preregistered R4 rule required zero resolved degradation across the 128-item sentiment, arithmetic, and language-model controls at each treatment seed's first recovery point.

All three recovered treatment seeds fail that strict gate because arithmetic shows a small but statistically resolved negative paired shift.

Seed 0:
- arithmetic delta: -0.01404
- CI95 [-0.02081,-0.00699]
- resolved degradation: yes
- sentiment delta: +0.03174, no resolved degradation
- LM mean-token-logprob delta: +0.00221, no resolved degradation

Seed 1:
- arithmetic delta: -0.01228
- CI95 [-0.01921,-0.00537]
- resolved degradation: yes
- sentiment delta: +0.02783, no resolved degradation
- LM delta: +0.00156, no resolved degradation

Seed 2:
- arithmetic delta: -0.01119
- CI95 [-0.01910,-0.00335]
- resolved degradation: yes
- sentiment delta: +0.02441, no resolved degradation
- LM delta: +0.00306, no resolved degradation

The arithmetic effect is small rather than catastrophic, but the frozen R4 contract intentionally fails on any resolved conventional-capability degradation. Therefore the correct disposition is:

`R4_GENERIC_DAMAGE`

No post-hoc tolerance relaxation is authorized.

## Interpretation

R4 demonstrates that the 4k pattern state can be driven back to GENERALIZING extremely quickly by full-model factual continuation, and that matched opposite-direction continuation drives the model further toward pattern matching.

However:
- full-model training can rapidly relearn or reconstruct the task;
- the first recovered state carries a small resolved arithmetic tradeoff;
- R4 is the weakest recovery tier for mechanism persistence.

Therefore R4 does not promote E3 persistence evidence.

Across R0-R4, the strongest supported classification remains:

`E2_ACCESSIBILITY_SHIFT_SUPPORTED`

with an additional bounded result:

`R3_PARAMETER_LIGHT_RECOVERY__PERSISTENCE_UNRESOLVED`.

WP06 should now close without an E3 or E4 claim.
