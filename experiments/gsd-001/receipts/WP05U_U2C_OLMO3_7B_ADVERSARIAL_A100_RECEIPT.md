# GSD-WP05U U2C — OLMo3 7B adversarial endpoint-confirmation receipt

Disposition: `PROMPT_SENSITIVE_UNCONFIRMED`

The frozen 7B candidate does **not** become a second `CONFIRMED_TRANSITION`.

## Hosted execution

- experiment: `GSD-WP05U-U2C-OLMO3-7B-ADVERSARIAL-A100-001`
- run: `20261008T123525Z-7752`
- source commit: `f3ddbb5028a8df2d26c387bcfd82c4ab62ad193a`
- hardware: NVIDIA A100-SXM4-40GB
- engineering status: `GREEN_ENGINEERING`
- source payload SHA-256: `afef9685cdf8b6b042d11b30fb49dfdc81ab3840dda2b6f03eb44c027beb8ee3`
- job SHA-256: `d7a0ad179ee212fc84598e358fe2385eede2397d7dcdf176f88f5ae9c8f6db81`
- host result SHA-256: `96008af836fc0dc7bb33ebf54ff0863087d78c5a2fb0643b452d2456b16b4955`
- scientific result SHA-256: `9fc1b53ed10e5de90bab5195a3f721241c2dbaca987c0ebe9b544765d4414573`
- protocol: `experiments/gsd-001/WP05U_7B_CONFIRM_ADVERSARIAL_PROTOCOL.md`

Inherited U2B conventional-capability gate: pass.

Frozen endpoints:
- `stage1-step565000` = PATTERN_MATCHING under canonical base replay;
- `stage1-step705000` = GENERALIZING under canonical base replay.

No checkpoint strictly inside the frozen interval was inspected before adjudication.

## Exact-boundary gate

Pass at both endpoints.

All scored continuations used an exact/stable forced-separate-continuation boundary mode.

## Independent replay

The canonical endpoint reversal reproduced for every preregistered seed:

- seed 0: PATTERN_MATCHING → GENERALIZING
- seed 1: PATTERN_MATCHING → GENERALIZING
- seed 2: PATTERN_MATCHING → GENERALIZING
- seed 3: PATTERN_MATCHING → GENERALIZING

Independent replay support: **4/4**.

At 565k the per-seed surprising-truth mean margins were approximately:
- -0.6389
- -0.6288
- -0.7168
- -0.5959

At 705k they were approximately:
- +0.3966
- +0.4453
- +0.3722
- +0.4405

## Prompt-variant falsification

The preregistered requirement was that every prompt variant preserve the endpoint states PATTERN_MATCHING → GENERALIZING.

Observed:

- `double_newline`: PATTERN_MATCHING → GENERALIZING — pass
- `expanded_markers`: PATTERN_MATCHING → UNCERTAIN — fail at 705k
- `instruction_prefix`: UNCERTAIN → GENERALIZING — fail at 565k

Therefore the frozen confirmation contract fails on prompt robustness even though exact boundaries, conventional controls, and 4/4 independent base replay pass.

Correct disposition:

`PROMPT_SENSITIVE_UNCONFIRMED`

## Replay-support accounting repair

The original scientific result JSON reported `replay_support: 0` because the shared validator returned early on prompt failure before counting already-recorded replay successes.

That number was a reporting/accounting defect, not a scientific failure and not a model-evaluation defect. The raw result itself records all four seed pairs as PATTERN_MATCHING → GENERALIZING.

The validator was repaired to compute replay support before the prompt-risk return:
- repair commit: `045ed50b67c3b10b67fa030821e5083c594b8a53`
- regression-test commit / exact repaired head: `4796e2a013f80a67d23f47fa877564171e4a5505`
- full test suite on repaired head: 72 passed

The repaired accounting is 4/4 and leaves the scientific disposition unchanged:
`PROMPT_SENSITIVE_UNCONFIRMED`.

## Scientific interpretation

This is useful negative evidence.

The 7B model exhibits a large, conventional-control-preserving, exact-boundary, seed-stable state reversal on the canonical prompt surface. But the reversal is not invariant to the preregistered prompt perturbations.

Therefore it cannot serve as the independent transition required to test transfer of the 1B layer-13 / Q14-H2 WP05 marker.

No mechanistic, cross-transition-predictive, E3, or E4 promotion is authorized.

The programme-level WP05 obligation remains:

`VALIDATE_FROZEN_LAYER13_Q14H2_SIGNATURE_ON_NEXT_INDEPENDENT_CONFIRMED_TRANSITION`.

The next utility-expansion frontier should not mine the already-seen 7B coarse trajectory post hoc. It should proceed through an independently preregistered transition source and/or the already-unlocked WP07 data-attribution lane.
