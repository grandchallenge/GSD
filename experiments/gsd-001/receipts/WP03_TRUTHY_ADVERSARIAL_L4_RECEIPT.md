# GSD-WP03 — Truthy adversarial validation receipt

Disposition: `TOKEN_BOUNDARY_ARTEFACT_RISK__NO_CONFIRMED_TRANSITION`

Run: `GSD-WP03-TRUTHY-ADVERSARIAL-L4-001` / `20261005T063131Z-31`
Source commit: `6bdc58fa3564408227039e259501d072a3e10c8c`
Hardware: NVIDIA L4
Status: `GREEN_ENGINEERING`
Revisions/summaries/manifests: 5/5/5
Payload SHA-256: `6154f0ef6cd44fca9bf0e0fab8e3749467fcddc6bc7feecfa246e555798b6a9e`
Job SHA-256: `296c94a93d205c1b93e5c17df1d6fedcffa92b91eb9c960058ad4260b7657db5`
Result SHA-256: `41b56530a09f5cf68816a6f9de22eacfa5739aa6c9636daa952d39ee64e81e3c`

Frozen protocol: `experiments/gsd-001/WP03_PROTOCOL.md`.

## Candidate 1 — surprising_truth 0 -> 2000

Expected: PATTERN_MATCHING -> GENERALIZING.

- Independent replay: 4/4 seeds preserve direction.
- double_newline preserves direction.
- expanded_markers gives GENERALIZING -> GENERALIZING.
- instruction_prefix gives GENERALIZING -> GENERALIZING.
- Boundary gate fails: all scored spans use `fallback_prompt_only`.

Disposition: `TOKEN_BOUNDARY_ARTEFACT_RISK`.

## Candidate 2 — surprising_truth 2000 -> 4000

Expected: GENERALIZING -> PATTERN_MATCHING.

- Independent replay: 4/4 seeds preserve direction.
- All 3 prompt attacks preserve direction.
- Boundary gate fails: all scored spans use `fallback_prompt_only`.

Disposition: `TOKEN_BOUNDARY_ARTEFACT_RISK`.

This is the strongest surviving behavioural signal, but it cannot be promoted under the frozen protocol.

## Candidate 3 — common_misconception 0 -> 2000

Expected: GENERALIZING -> PATTERN_MATCHING.

- Independent replay: 4/4 seeds preserve direction.
- double_newline and instruction_prefix preserve direction.
- expanded_markers gives UNCERTAIN -> PATTERN_MATCHING.
- Boundary gate fails: all scored spans use `fallback_prompt_only`.

Disposition: `TOKEN_BOUNDARY_ARTEFACT_RISK`.

## Flanking localization

Pooled base replay:
- surprising_truth: step 0 PATTERN_MATCHING; 1000/2000 GENERALIZING; 3000/4000 PATTERN_MATCHING.
- common_misconception: step 0 GENERALIZING; 1000/2000/3000/4000 PATTERN_MATCHING.

The apparent reversals localize to 0-1000, 2000-3000, and 0-1000 respectively.

## Boundary diagnosis

The failure is systematic. Across both tasks, all four replay seeds, prompt attacks, and implicated endpoints, every correct/incorrect answer span used the fallback boundary path. No current Truthy result can therefore receive `CONFIRMED_TRANSITION`.

Next action: define an exact continuation-token boundary contract, add regression tests, freeze a boundary-repair replay, and rerun before any G1 promotion.

Claim boundary: behavioural evidence only; no mechanism, persistence, capacity-allocation, optimizer, or architecture claim.
