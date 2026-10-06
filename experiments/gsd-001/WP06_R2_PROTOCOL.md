# GSD-WP06-R2 — Sparse causal recovery

Status: frozen before execution.

Parent work package: `GSD-WP06 — Recovery and Residual test`.

Predecessors:
- R0: `R0_INSTRUCTION_ACCESSIBILITY`, supporting E2 accessibility shift.
- R1: `R1_DISCOVERY_ONLY_REJECTED`; no E3 promotion.
- R1 post-hoc prohibition remains binding: do not test the R1 layer-16 whole-residual discovery result on the consumed rows 64–127.

Confirmed transition:
- task: `truthy_answer/surprising_truth`
- source: `stage1-step2000-tokens5B` = GENERALIZING
- flank: `stage1-step3000-tokens7B` = PATTERN_MATCHING
- target: `stage1-step4000-tokens9B` = PATTERN_MATCHING
- context: original seed-0 `base_k8` misleading demonstrations
- stable-control pair 2k -> 4k: pass
- exact continuation-token boundary: required

## Fresh evaluation data

The source dataset contains 287 test rows.

Previously consumed:
- rows 0–127.

R2 uses only never-used rows:
- discovery: rows 128–191;
- held-out confirmation: rows 192–255.

Reserved and untouched after R2:
- rows 256–286.

No R2 module selection may use held-out or reserve rows.

## Sparse intervention class

OLMo2 decoder layers add separately normalized attention and MLP contributions:

- `post_attention_layernorm(self_attn(...))`;
- `post_feedforward_layernorm(mlp(...))`.

R2 patches exactly one such contribution at answer-prediction positions.

Frozen candidate modules:
- layer 12 attention contribution
- layer 12 MLP contribution
- layer 13 attention contribution
- layer 13 MLP contribution
- layer 14 attention contribution
- layer 14 MLP contribution
- layer 15 attention contribution
- layer 15 MLP contribution
- layer 16 attention contribution
- layer 16 MLP contribution

For a candidate module:
1. run source checkpoint 2k on the exact teacher-forced token sequence;
2. capture that module's post-normalized contribution only at answer-prediction positions;
3. run target checkpoint on the same token sequence;
4. replace only the corresponding target module contribution at those positions;
5. leave every other target activation and all parameters unchanged.

No whole-residual patch is allowed in R2.

## Selection rule

A discovery candidate qualifies only if:
- source 2k is GENERALIZING on discovery rows;
- target 4k is PATTERN_MATCHING on discovery rows;
- the matched source->target sparse patch makes target 4k GENERALIZING;
- exact answer-boundary gate passes.

Select the first qualifying candidate under this fixed order:
1. increasing layer number 12 -> 16;
2. attention before MLP within a layer.

If no candidate qualifies:
`NO_R2_RECOVERY`.

## Fresh held-out confirmation

For the selected sparse module, all must hold on rows 192–255:

1. source 2k base state = GENERALIZING;
2. target 4k base state = PATTERN_MATCHING;
3. matched 2k->4k sparse patch = GENERALIZING;
4. flank 3k base state = PATTERN_MATCHING;
5. matched 2k->3k sparse patch = GENERALIZING;
6. target->target identity sparse patch preserves the 4k base state;
7. identity max per-item margin drift <= 1e-3;
8. seed-17 item-shuffled source sparse patch does not produce GENERALIZING;
9. inherited broad stable controls remain true;
10. exact answer-boundary contract passes.

Shuffling preserves correct/incorrect answer class and answer-token count.

## Intervention cost

`C(delta) = (modules_patched, residual_vectors_patched, hidden_dimensions)`.

For a successful R2 intervention:
- modules_patched = 1;
- hidden_dimensions = 2048;
- residual_vectors_patched = number of answer-prediction positions.

This is strictly sparser in module count than R1 whole-residual replacement.

## Dispositions

- `NO_R2_RECOVERY`
- `R2_DISCOVERY_ONLY_REJECTED`
- `R2_IMPLEMENTATION_CONTROL_FAILED`
- `R2_NONSPECIFIC_MODULE_EFFECT`
- `R2_SPARSE_CAUSAL_RECOVERY`

Evidence interpretation:
- `R2_SPARSE_CAUSAL_RECOVERY` supports E3
  `MECHANISM_PERSISTENCE_SUPPORTED` under the programme definition.
- It does not prove literal circuit identity.
- Any weaker disposition leaves the strongest programme result at E2 unless separate evidence warrants otherwise.

No parameter edit, training update, optimizer claim, architecture claim, or capacity-allocation claim is authorized by R2.
