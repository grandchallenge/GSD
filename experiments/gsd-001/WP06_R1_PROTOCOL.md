# GSD-WP06-R1 — Activation-space recovery

Status: frozen before execution.

Parent work package: `GSD-WP06 — Recovery and Residual test`.

Confirmed transition:
- task: `truthy_answer/surprising_truth`
- source: `stage1-step2000-tokens5B` = GENERALIZING
- localization flank: `stage1-step3000-tokens7B` = PATTERN_MATCHING
- primary target: `stage1-step4000-tokens9B` = PATTERN_MATCHING
- original context: seed-0 `base_k8` misleading demonstrations
- inherited broad stable-control pair 2k -> 4k: pass
- R0 result: `R0_INSTRUCTION_ACCESSIBILITY`

Purpose:
test whether a bounded internal intervention from the 2k generalizing checkpoint
can restore generalizing behaviour at the later checkpoints while the original
misleading demonstration context remains unchanged.

Model geometry:
- 16 decoder layers;
- hidden size 2048;
- patch points are residual-stream states after source layers 4, 8, 12, and 16.

## Intervention

For each teacher-forced correct/incorrect answer sequence, patch only the
residual vectors at positions that predict the answer tokens.

At patch point `p`:
1. run the 2k source checkpoint on the exact token sequence;
2. capture the residual stream after source layer `p`;
3. run the target checkpoint on the same exact token sequence;
4. replace only the target residual vectors at answer-prediction positions
   with the item-matched source vectors;
5. continue the remaining target computation unchanged.

Patch semantics:
- p=4 patches the input to target layer 5;
- p=8 patches the input to target layer 9;
- p=12 patches the input to target layer 13;
- p=16 patches the input to the target final RMSNorm.

No parameters are changed. No prompt tokens, demonstration order, or answer
tokens are changed.

## Recovery cost

Predeclared intervention cost is the tuple

`C(delta) = (source_layers_imported, residual_vectors_patched, hidden_dimensions)`.

For this model:
- `source_layers_imported = p`;
- `hidden_dimensions = 2048`;
- `residual_vectors_patched` is the number of answer-prediction positions.

Among discovery layers that recover, select the **earliest** patch point.
This minimizes imported source computation and leaves the largest target
suffix intact.

## Discovery / confirmation split

Use the same first 128 Truthy test rows as prior work.

- discovery: rows 0-63;
- held-out confirmation: rows 64-127.

Layer selection is allowed only on the discovery half.

Frozen patch grid:
- 4
- 8
- 12
- 16

A discovery layer qualifies only if:
- source 2k base state is GENERALIZING on discovery;
- target 4k base state is PATTERN_MATCHING on discovery;
- matched source->target patch makes target 4k GENERALIZING on discovery;
- exact answer-boundary contract passes.

Select the earliest qualifying layer.

## Held-out confirmation

The selected layer is confirmed only if all hold:

1. source 2k is GENERALIZING on held-out rows;
2. unpatched target 4k is PATTERN_MATCHING on held-out rows;
3. matched 2k->4k patch makes 4k GENERALIZING on held-out rows;
4. unpatched flank 3k is PATTERN_MATCHING on held-out rows;
5. matched 2k->3k patch makes 3k GENERALIZING on held-out rows;
6. target->target identity patch preserves the unpatched 4k state and has
   maximum per-item margin drift <= 1e-3;
7. a fixed seed-17 item-shuffled 2k->4k patch does **not** produce
   GENERALIZING on held-out rows;
8. inherited broad stable controls remain true;
9. exact continuation-token boundary contract passes.

The shuffled control preserves correct/incorrect answer class and answer-token
count while permuting source residuals across items.

## Dispositions

- no discovery layer recovers:
  `NO_R1_RECOVERY`
- discovery recovery fails held-out:
  `R1_DISCOVERY_ONLY_REJECTED`
- identity sham fails:
  `R1_IMPLEMENTATION_CONTROL_FAILED`
- shuffled patch also recovers:
  `R1_NONSPECIFIC_ACTIVATION_EFFECT`
- matched patch passes all held-out gates:
  `R1_ACTIVATION_RECOVERY`

Evidence interpretation:
- `R1_ACTIVATION_RECOVERY` supports E3
  `MECHANISM_PERSISTENCE_SUPPORTED` under the programme definition because a
  small cross-checkpoint activation substitution causally restores behaviour
  under the original hostile context.
- It still does not prove literal circuit identity.
- Any weaker disposition remains at E2 or below.

No mechanism, optimizer, capacity-allocation, or architecture claim is
authorized beyond this declared evidence ladder.
