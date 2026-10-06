# GSD-WP05U — Functionalization and utility-expansion protocol

Status: frozen before execution.

Parent: `GSD-WP05 — Mechanistic/operator localization`.

Purpose: turn the validated within-transition WP05 signature into a reusable object without weakening the programme-level held-out-transition gate.

Frozen inherited signature:
- representation: transformer layer 13 residual representation;
- operator: layer 14 query projection, head 2;
- discovery/evaluation direction agreement: 0.99550 / 0.99752;
- operator evaluation locality percentile: 0.99609;
- strongest programme evidence before this tranche: E2 `ACCESSIBILITY_SHIFT_SUPPORTED`.

## Claim firewall

This tranche does not create a second independent transition.

All `truthy_answer/surprising_truth` rows used here were previously consumed by GSD. Therefore:
- no programme-fresh holdout claim;
- no cross-transition prediction claim;
- no universal "head 2" identity claim;
- no E3/E4 promotion from a same-transition intervention alone.

The programme-level WP05 exit obligation remains:
`VALIDATE_FROZEN_LAYER13_Q14H2_SIGNATURE_ON_NEXT_INDEPENDENT_CONFIRMED_TRANSITION`.

## Utility-expansion lanes

### U1 — directional causal intervention (execute now)

Question: does the frozen layer-14 query-head-2 transition direction merely correlate with the behavioural flip, or does moving the target checkpoint along that direction move behaviour?

Discovery rows: 0-63.
Evaluation rows: 64-127.

Direction construction:
1. At the final prompt token, after native q normalization, compute the discovery mean query vector for each frozen candidate at 2k and 3k.
2. Define transition direction `delta = mean_3k - mean_2k`.
3. Freeze the vectors before scoring the evaluation partition.

Target intervention:
- checkpoint: 4k target;
- target candidate: Q / layer 14 / head 2;
- `reverse`: add `-1.0 * delta` at the final prompt token;
- `amplify`: add `+1.0 * delta`.

Structural controls, each using its own discovery transition direction and the same `-1.0` intervention:
- Q / layer 14 / head 1;
- Q / layer 14 / head 3;
- Q / layer 13 / head 2;
- Q / layer 15 / head 2.

Outcome:
- paired change in exact-boundary behavioural margin relative to unmodified 4k;
- bootstrap 95% CI of the paired change;
- state classification for every intervention.

Directional causal support requires all of:
1. target reverse mean margin delta >= +0.25;
2. target amplify mean margin delta <= -0.25;
3. target reverse mean effect exceeds every structural-control reverse mean effect by >= 0.10.

Dispositions:
- `HEAD2_DIRECTIONAL_CAUSAL_SUPPORT`
- `HEAD2_NONSPECIFIC_CAUSAL_EFFECT`
- `NO_HEAD2_CAUSAL_EFFECT`
- `EXACT_BOUNDARY_CONTRACT_FAILED`

A positive U1 result is causal support for this intervention surface on this transition, not mechanism identity.

### U2 — head identity versus functional subspace

On the next independent confirmed transition and on targeted 7B replication, test in this order:
1. literal same coordinate, Q/L14/H2 where architecture permits;
2. normalized depth coordinate around 14/16 = 0.875;
3. best matching query change direction by cosine/subspace angle without using behavioural labels;
4. low-dimensional late-query subspace learned only from prior transitions.

The scientific object should be promoted from a literal head number to a functional subspace only if transfer improves under a preregistered held-out transition test.

### U3 — temporal precursor scan

When checkpoint density permits, evaluate the frozen signature before, during, and after a transition window.

A useful monitoring diagnostic must move before or at the behavioural transition, not only after it.

No early-warning claim is permitted from three coarse checkpoints.

### U4 — data attribution coupling

GSD-WP07 should treat the frozen WP05 directions as dependent variables in addition to behavioural margin.

For each candidate data class or micro-continuation:
- measure behavioural state movement;
- measure layer-13 representation movement;
- measure the late-query signature/subspace movement;
- require matched conventional-capability controls.

The preferred mechanistic chain is:
`training data -> internal operator movement -> behavioural state movement`.

### U5 — scale/seed portability

Targeted OLMo3 7B/32B replication should test relative depth/subspace portability, not assume literal head-number portability.

No scale result may be used to satisfy WP05's held-out-transition gate unless it independently meets the existing `CONFIRMED_TRANSITION` contract.

## Promotion ladder

The reusable object may be renamed only as evidence accumulates:

1. `Q14H2_WITHIN_TRANSITION_MARKER` — current.
2. `Q14H2_DIRECTIONAL_CAUSAL_SURFACE` — only if U1 passes.
3. `LATE_QUERY_TRANSITION_SUBSPACE` — only if U2 transfers beyond literal identity.
4. `GSD_STATE_MONITOR` — only if U3 predicts held-out transition timing.
5. `GSD_CONTROL_COORDINATE` — only if WP07/WP08 interventions move occupancy under matched capability.

This naming ladder is a claim firewall, not a target to optimize toward.
