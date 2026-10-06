# GSD-WP06-R3C — Conventional-control gate repair

Status: frozen before execution.

Purpose:
repair only the ordinary-capability control gate that made WP06-R3 terminate as
`R3_BASELINE_UNRESOLVED`.

The primary R3 reserve result is immutable and is not rerun or retuned here.

## Why a repair is required

R3's primary intervention behaved strongly and specifically on the untouched
Truthy reserve:
- 3/3 factual-treatment adapters = GENERALIZING;
- 0/3 reversed-label controls = GENERALIZING.

However the frozen 32-item arithmetic control was already `UNCERTAIN` before
adaptation. The absolute-state requirement was therefore underpowered and
inconsistent with the programme's earlier stable-control contract, which is
based on **paired degradation**, not on every baseline control being
individually GENERALIZING.

R3C does not relax the primary recovery criterion. It only repairs this control
measurement.

## Adapter reproduction

Reproduce only the three factual-treatment adapters using the exact R3 recipe:
- target checkpoint: `stage1-step4000-tokens9B`;
- module: layer-12 `self_attn.o_proj`;
- rank 1, alpha 1.0;
- 4096 trainable parameters;
- training rows 128-191;
- 32 two-item steps;
- AdamW lr 3e-3;
- seeds 0,1,2.

No Truthy reserve scoring is permitted.

Reproduction sanity:
- each final pairwise training loss must be < 1e-3;
- each adapter delta Frobenius norm must lie within 5% of the corresponding
  original R3 value:
  - seed 0: 4.3695998192
  - seed 1: 4.4854421616
  - seed 2: 4.4150676727

## Conventional controls

Evaluate the unadapted target and every reproduced treatment adapter on the
same 128-item conventional suite used by the programme:

1. ordinary SST-2 sentiment with reconstructed correct-label demonstrations;
2. conventional zero-shot arithmetic;
3. held-out SST-2 review mean token log-likelihood.

For sentiment and arithmetic, use exact continuation-token scoring and paired
soft-margin vectors.

For language modelling, use per-text mean next-token log-probability.

## Repair gate

For each treatment seed and each control family, compute the paired
adapter-minus-base bootstrap delta CI on the identical 128 examples.

A family shows resolved degradation iff the upper endpoint of the 95% CI is
strictly below zero.

A reproduced adapter is `control_ok` only if **zero of three** families show
resolved degradation.

R3C passes only if:
- exact answer boundaries pass;
- all three adapter reproductions pass the norm/loss sanity checks;
- all three reproduced adapters have `control_ok: true`.

Dispositions:
- `R3_CONTROL_GATE_REPAIRED`
- `R3_CONTROL_GATE_FAILED`
- `R3_ADAPTER_REPRODUCTION_FAILED`
- `EXACT_BOUNDARY_CONTRACT_FAILED`

## Combined interpretation

If R3C passes, the immutable R3 primary result may be summarized as:

`R3_PARAMETER_LIGHT_RECOVERY__PERSISTENCE_UNRESOLVED`

This means:
- a 4096-parameter, 32-step adapter robustly restored the generalizing behaviour
  on the previously untouched reserve;
- matched reversed-label controls moved behaviour in the opposite direction;
- independently reproduced treatment adapters do not show resolved degradation
  on the conventional capability suite.

It still does **not** prove persistence of the same mechanism and does not
exclude rapid relearning.

No E3 promotion follows from R3/R3C alone.

Rows 256-286 must not be evaluated in R3C.
