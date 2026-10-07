# GSD-WP05U U2C — OLMo3 7B adversarial endpoint confirmation

Status: frozen before execution.

Parent discovery:
`experiments/gsd-001/receipts/WP05U_U2A_OLMO3_7B_COARSE_A100_RECEIPT.md`

Control gate:
`experiments/gsd-001/receipts/WP05U_U2B_OLMO3_7B_CONTROLS_A100_RECEIPT.md`

Frozen candidate:
- model: `allenai/Olmo-3-1025-7B`
- task: `truthy_answer/surprising_truth`
- left: `stage1-step565000` = PATTERN_MATCHING
- right: `stage1-step705000` = GENERALIZING
- paired conventional controls: PASS

No checkpoint strictly inside 565000→705000 may be inspected until this endpoint test is adjudicated.

## Confirmation test

Evaluate both endpoints under the exact-continuation boundary contract using:
- base replay seeds: 0, 1, 2, 3;
- prompt variants at seed 0:
  - `instruction_prefix`
  - `expanded_markers`
  - `double_newline`
- max evaluation rows: 128;
- BF16 on A100.

Confirmation requires:
1. all scored answer continuations use an exact/stable boundary mode;
2. every prompt variant has endpoint states PATTERN_MATCHING→GENERALIZING;
3. at least 3 of 4 base replay seeds have endpoint states PATTERN_MATCHING→GENERALIZING;
4. inherited U2B paired control gate remains true.

Disposition:
- `CONFIRMED_TRANSITION`
- `TOKEN_BOUNDARY_ARTEFACT_RISK`
- `PROMPT_SENSITIVE_UNCONFIRMED`
- `INDEPENDENT_REPLAY_FAILED`
- `GENERIC_DEGRADATION_NOT_GSD`

## Claim boundary

A positive result creates a second independent confirmed transition, at 7B scale. It authorizes a frozen WP05 transfer test using the already-selected 1B signature.

It does not itself establish that the 1B layer-13/Q14H2 marker transfers, that any single head is causal, or that a universal scalar state diagnostic exists.
