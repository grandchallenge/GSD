# GSD-WP03R — Truthy exact-boundary adversarial validation

Disposition: `G1_OPEN__ONE_CONFIRMED_TRANSITION`

Hosted run:
- experiment: `GSD-WP03R-TRUTHY-EXACT-BOUNDARY-L4-001`
- run: `20261005T090911Z-396`
- source commit: `c7dccf39353a0c9094e18d1d475b82411b33f4af`
- hardware: NVIDIA L4
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 5/5/5
- payload SHA-256: `32d60414da81becc8f60b444ad5f279484013eb121cca50e68e03dd81f1d6724`
- job SHA-256: `b7fb9e3cb2573d7d272608dc6399a9044f9d918cbc31d7d72db04fbd580c7a64`
- result SHA-256: `3939a5fc5e2dd8dd8b7d5e50356baff4bc8c3712aa0f472a3b0110e264ec1d29`
- protocol: `experiments/gsd-001/WP03_BOUNDARY_REPAIR_PROTOCOL.md`

## Candidate 1 — surprising_truth 0 -> 2000

Expected:
`PATTERN_MATCHING -> GENERALIZING`

- stable controls: pass
- exact endpoint boundaries: pass
- independent replay endpoint states: 4/4 seeds preserve expected direction
- prompt variants:
  - double_newline: preserves expected direction
  - expanded_markers: GENERALIZING -> GENERALIZING
  - instruction_prefix: GENERALIZING -> GENERALIZING

Disposition:
`PROMPT_SENSITIVE_UNCONFIRMED`

## Candidate 2 — surprising_truth 2000 -> 4000

Expected:
`GENERALIZING -> PATTERN_MATCHING`

- stable controls: pass
- exact endpoint boundaries: pass
- independent replay: 4/4 seeds preserve expected direction
- prompt variants: 3/3 preserve expected direction
- required replay support: 3/4

Disposition:
`CONFIRMED_TRANSITION`

Flanking localization:
- step 2000: GENERALIZING
- step 3000: PATTERN_MATCHING
- step 4000: PATTERN_MATCHING

Thus the confirmed transition localizes to the interval between steps 2000 and 3000 under the pooled base replay.

## Candidate 3 — common_misconception 0 -> 2000

Expected:
`GENERALIZING -> PATTERN_MATCHING`

- stable controls: pass
- exact endpoint boundaries: pass
- independent replay endpoint states: 4/4 seeds preserve expected direction
- prompt variants:
  - double_newline: preserves expected direction
  - instruction_prefix: preserves expected direction
  - expanded_markers: UNCERTAIN -> PATTERN_MATCHING

Disposition:
`PROMPT_SENSITIVE_UNCONFIRMED`

## Gate result

Gate G1 requires at least one `CONFIRMED_TRANSITION` based on soft margins.

Gate G1 is therefore OPEN.

Unlocked:
- GSD-WP04 — CPS transition-local dynamics
- GSD-WP05 — mechanistic/operator localization
- GSD-WP06 — recovery and Residual test

Claim boundary:
this establishes one robust behavioural transition at OLMo2 1B under the frozen GSD protocol. It does not establish a mechanism, optimizer cause, persistence state, capacity-allocation explanation, or architecture-level conclusion.
