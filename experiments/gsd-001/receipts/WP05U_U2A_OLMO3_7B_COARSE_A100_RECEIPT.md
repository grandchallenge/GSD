# GSD-WP05U U2A — OLMo3 7B coarse transfer-scan receipt

Disposition: `7B_CANDIDATE_TRANSITION`

This is discovery evidence only. It is not a `CONFIRMED_TRANSITION`.

Hosted execution:
- experiment: `GSD-WP05U-U2A-OLMO3-7B-COARSE-A100-001`
- run: `20261006T112214Z-3745`
- source commit: `5494304838f6ea9784f68bdd1af254420c0401c5`
- hardware: NVIDIA A100-SXM4-40GB
- engineering status: `GREEN_ENGINEERING`
- source payload SHA-256: `d59e8b6cca0d37ff07837c70746bc8e8e46b31014e4a0f764a038c5533449c90`
- job SHA-256: `cf8b4586df611a6b96a74b394fee27d8977755837e1f14734fe1a7543a013682`
- host result SHA-256: `a18949e8043cdae77a7c87afbf986f568008de0599282356f4140c47d9a29c09`
- model: `allenai/Olmo-3-1025-7B`
- probe family: `truthy_answer`
- max evaluation rows per task: 64
- canonical seed count: 1

## Primary surprising-truth trajectory

| Stage-1 step | Mean soft margin | State |
|---:|---:|---|
| 0 | -1.15867 | PATTERN_MATCHING |
| 282000 | +2.60077 | GENERALIZING |
| 565000 | -0.40875 | PATTERN_MATCHING |
| 705000 | +0.72050 | GENERALIZING |
| 846000 | +0.36069 | GENERALIZING |
| 1130000 | +0.23061 | UNCERTAIN |
| 1413814 | +1.29332 | GENERALIZING |

The coarse scan therefore contains multiple resolved state reversals.

The narrowest preregistered sampled bracket for the primary task is:

`stage1-step565000 PATTERN_MATCHING -> stage1-step705000 GENERALIZING`

with endpoint bootstrap intervals:
- step 565000: [-0.68729,-0.12994]
- step 705000: [+0.42093,+1.00928]

This bracket is frozen before any finer checkpoint inspection.

## Secondary observation

`truthy_answer/common_misconception` also changes state across the coarse trajectory, including GENERALIZING at step 0 and PATTERN_MATCHING at step 282000. It is not used to replace the frozen primary bracket.

## Adjudication

The correct disposition is:

`7B_CANDIDATE_TRANSITION`

A coarse state reversal cannot satisfy the WP05 cross-transition gate.

Before any mechanistic/subspace transfer test, the frozen 565000→705000 candidate must pass:
1. matched conventional-capability controls;
2. exact continuation-boundary audit;
3. independent multi-seed replay;
4. all frozen prompt variants.

No intermediate checkpoint inside the bracket may be inspected before this endpoint confirmation is adjudicated.

If confirmed, it becomes an independent scale transition suitable for testing normalized-depth/late-query subspace transfer. If rejected, the negative confirmation is retained and no arbitrary post-hoc checkpoint search is authorized.
