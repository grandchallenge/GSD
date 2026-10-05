# GSD-WP01 — L4 checkpoint-harness receipt

Disposition: `WP01_CLOSED__20_CHECKPOINTS_REPLAYED`

Hosted run:
- experiment: `GSD-WP01-OLMO2-1B-CRT-COARSE20-L4-001`
- source commit: `db4abac3bc6bdeeafdd0698ab72098ff5902baa8`
- payload SHA-256: `4254f3cc96cc6977da46f0545f3c85f768424dd171e1a80243e707466c8e1199`
- hardware: NVIDIA L4
- status: `GREEN_ENGINEERING`
- summaries: 20/20
- manifests: 20/20
- started: 2026-10-05T00:52:24.626898Z
- finished: 2026-10-05T01:25:28.897763Z

Probe:
- task: `intuitive_answer/crt`
- max_eval: 128
- checkpoint design: steps 0, 2000, ..., 36000 plus step 35000 terminal flank
- exact checkpoint count: 20

WP02 catalogue result for this probe:

```json
{
  "candidate_transitions": [],
  "checkpoint_count": 20,
  "control_evidence_provided": false,
  "state_reversals": [],
  "target": "intuitive_answer/crt",
  "target_type": "task"
}
```

Observed state sequence:
- step 0: PATTERN_MATCHING
- steps 2000 and 4000: UNCERTAIN
- steps 6000 through 36000: PATTERN_MATCHING

Therefore CRT supplies no confident generalizing-to-pattern-matching or pattern-matching-to-generalizing reversal at this scale. This is a negative probe result, not evidence against the broader programme.

Claim boundary: behavioural evidence only; no mechanism claim.
