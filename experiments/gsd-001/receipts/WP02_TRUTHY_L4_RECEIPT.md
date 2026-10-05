# GSD-WP02 — Truthy Answer L4 receipt

Disposition: `STATE_REVERSALS_FOUND__CONTROL_VALIDATION_REQUIRED`

Hosted run:
- experiment: `GSD-WP02-OLMO2-1B-TRUTHY-COARSE20-L4-001`
- source commit at execution: `6bfb50f5d3db3804388dbc2ec17c2f541e734293`
- payload SHA-256: `d48ffcf5fe680e59d633202e4f42e134cf1b5b0d28f45b78457ae3a973aae1ac`
- job SHA-256: `e7a55f29cd2edfa799fc4bf54bf3a7346db2acc7366ed09c219af3f0dfd6fbca`
- result SHA-256: `9d18395bf03c08acdac8c5423d25fb1c1cd54db314553de4425ad591b4c32c8d`
- hardware: NVIDIA L4
- status: `GREEN_ENGINEERING`
- checkpoints: 20/20
- summaries: 20/20
- started: 2026-10-05T02:26:36.946668Z
- finished: 2026-10-05T03:06:49.676963Z

Task-level transition catalogue:

## truthy_answer/surprising_truth

State sequence begins:

- step 0: `PATTERN_MATCHING`, mean margin -0.334778, CI95 [-0.458698, -0.206360]
- step 2000: `GENERALIZING`, mean margin +1.169246, CI95 [+1.102613, +1.233366]
- step 4000: `PATTERN_MATCHING`, mean margin -2.247805, CI95 [-2.331673, -2.163301]

State reversals:
1. step 0 -> 2000: PATTERN_MATCHING -> GENERALIZING
2. step 2000 -> 4000: GENERALIZING -> PATTERN_MATCHING

Both are `STATE_REVERSAL_REQUIRES_CONTROL_VALIDATION`.

## truthy_answer/common_misconception

State sequence begins:

- step 0: `GENERALIZING`, mean margin +1.137590, CI95 [+1.073981, +1.199302]
- step 2000: `PATTERN_MATCHING`, mean margin -2.140625, CI95 [-2.196777, -2.085449]

State reversal:
1. step 0 -> 2000: GENERALIZING -> PATTERN_MATCHING

Disposition: `STATE_REVERSAL_REQUIRES_CONTROL_VALIDATION`.

No transition is promoted to `CANDIDATE_TRANSITION` because explicit stable-control evidence has not yet been supplied for the endpoint checkpoints.

Claim boundary: behavioural evidence only; no mechanism claim.
