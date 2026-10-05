# GSD-WP03 — Truthy boundary-repair validation protocol

Status: frozen before V2 execution.

Predecessor: `WP03_PROTOCOL.md`.

The V1 replay terminated at `TOKEN_BOUNDARY_ARTEFACT_RISK` because every
Truthy answer span used the unsafe `fallback_prompt_only` path. V1 remains
immutable and cannot support a confirmed transition.

## Exact continuation boundary V2

For each prompt/answer pair:

1. If tokenizing `prompt + " "` is a true prefix of the canonical full-text
   tokenization, use that canonical token sequence and mark
   `stable_prompt_space`.
2. Otherwise tokenize `prompt` and `" " + answer` separately, concatenate
   those token IDs, and mark `forced_separate_continuation`.
3. Pass the returned token IDs directly to vLLM through `TokensPrompt`.
   Re-tokenization of the full text is prohibited.
4. A span is exact only if its mode is one of those two modes. Any unresolved
   mode yields `TOKEN_BOUNDARY_ARTEFACT_RISK`.

The forced mode defines the conditional continuation score on the model's
token sequence with an explicit prompt/answer boundary. It is a new scoring
contract, not a reinterpretation of V1.

## Candidates

1. surprising_truth: step 0 PATTERN_MATCHING -> step 2000 GENERALIZING.
2. surprising_truth: step 2000 GENERALIZING -> step 4000 PATTERN_MATCHING.
3. common_misconception: step 0 GENERALIZING -> step 2000 PATTERN_MATCHING.

## Replay

- checkpoints: 0, 1000, 2000, 3000, 4000;
- seeds: 0, 1, 2, 3;
- 128 fixed test rows per task;
- prompt attacks: instruction_prefix, expanded_markers, double_newline;
- stable controls inherited from WP02;
- at least 3/4 seeds must preserve the candidate endpoint states;
- all three prompt attacks must preserve the endpoint states;
- flanks are supplementary only.

Terminal dispositions remain:
`CONFIRMED_TRANSITION`,
`TOKEN_BOUNDARY_ARTEFACT_RISK`,
`PROMPT_SENSITIVE_UNCONFIRMED`,
`INDEPENDENT_REPLAY_FAILED`,
or `GENERIC_DEGRADATION_NOT_GSD`.

Claim boundary: behavioural transition validation only. No mechanism,
persistence, capacity-allocation, optimizer, or architecture claim.
