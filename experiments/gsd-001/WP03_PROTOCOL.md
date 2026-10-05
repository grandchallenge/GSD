# GSD-WP03 — Truthy adversarial transition validation protocol

Status: frozen before WP03 execution.

Candidates under test:

1. `truthy_answer/surprising_truth`: step 0 PATTERN_MATCHING -> step 2000 GENERALIZING.
2. `truthy_answer/surprising_truth`: step 2000 GENERALIZING -> step 4000 PATTERN_MATCHING.
3. `truthy_answer/common_misconception`: step 0 GENERALIZING -> step 2000 PATTERN_MATCHING.

Exact model revisions:
- step 0
- step 1000
- step 2000
- step 3000
- step 4000

Base replay:
- 4 deterministic demo seeds: 0, 1, 2, 3;
- 128 fixed test rows per task;
- upstream random-demo semantics retained;
- sum log-probability margin remains primary.

Prompt attacks, evaluated at candidate endpoints with seed 0:
1. `instruction_prefix`: add a neutral True/False task instruction.
2. `expanded_markers`: replace structural `Q:` / `A:` markers by `Question:` / `Answer:`.
3. `double_newline`: change only the block joiner from one newline to two.

Token-boundary audit:
- every correct and incorrect answer span must use `stable_prompt_space`;
- any fallback boundary on an endpoint yields `TOKEN_BOUNDARY_ARTEFACT_RISK`.

Independent replay gate:
- at least 3 of 4 seeds must classify both endpoints confidently in the original candidate direction.

Prompt robustness gate:
- each of the three attack variants must classify both endpoints confidently in the original candidate states.

Stable-control gate:
- inherited from `WP02_STABLE_CONTROLS_L4_RECEIPT.md`;
- both relevant checkpoint pairs already have `control_ok: true`.

Flanking evidence:
- steps 1000 and 3000 are reported to localize persistence/transition timing;
- flanking evidence is supplementary and cannot rescue a failed independent replay or prompt-robustness gate.

Terminal dispositions:
- `CONFIRMED_TRANSITION`
- `TOKEN_BOUNDARY_ARTEFACT_RISK`
- `PROMPT_SENSITIVE_UNCONFIRMED`
- `INDEPENDENT_REPLAY_FAILED`
- `GENERIC_DEGRADATION_NOT_GSD`

Claim boundary:
WP03 can confirm a behavioural transition. It cannot establish a mechanism, persistence, capacity-allocation cause, optimizer cause, or architectural cause.
