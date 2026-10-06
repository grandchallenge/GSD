# GSD-WP06-R3 — Parameter-light recovery

Status: frozen before execution.

Parent work package: `GSD-WP06 — Recovery and Residual test`.

Predecessor evidence:
- R0: `R0_INSTRUCTION_ACCESSIBILITY` — E2 accessibility/expression supported.
- R1: `R1_DISCOVERY_ONLY_REJECTED`.
- R2: `NO_R2_RECOVERY`.
- No E3 persistence promotion is currently authorized.

Confirmed transition:
- task: `truthy_answer/surprising_truth`
- source checkpoint: `stage1-step2000-tokens5B` = GENERALIZING
- target checkpoint: `stage1-step4000-tokens9B` = PATTERN_MATCHING
- context: original seed-0 `base_k8` misleading demonstrations.

## Intervention class

R3 uses exactly one rank-1 LoRA adapter on the target checkpoint:

- module: decoder layer 12 `self_attn.o_proj`;
- rank: 1;
- alpha: 1.0;
- dropout: 0;
- base parameters frozen;
- expected trainable parameter count: 4096;
- no bias update;
- no other module is trainable.

Layer 12 attention is fixed before R3 because it was the strongest directional
single-module R2 intervention. No R3 layer/module search is permitted.

## Training data and budget

Training uses only already-consumed Truthy rows 128–191.

The prompt is the original seed-0 `base_k8` misleading-demonstration context.

Frozen budget per adapter:
- 64 training items;
- batch size: 2 items = 4 teacher-forced answer sequences;
- one epoch exactly;
- 32 optimizer steps;
- optimizer: AdamW;
- learning rate: 3e-3;
- weight decay: 0;
- gradient norm clip: 1.0;
- pairwise objective: `softplus(-(logP(correct)-logP(incorrect)))`.

Seeds:
- 0
- 1
- 2

Each seed changes only LoRA initialization and training-item order.

## Matched rapid-relearning / nonspecific-update control

For every treatment seed, train a second adapter from the identical frozen
target checkpoint with the identical module, rank, optimizer, learning rate,
step count, and data order, but reverse the pairwise objective so that the
adapter is trained to prefer the predefined incorrect/pattern answer.

Thus there are:
- 3 factual/generalizing treatment adapters;
- 3 reversed-label control adapters.

A treatment effect is not accepted if reversed-label controls also produce
GENERALIZING reserve behaviour.

## Untouched confirmation set

Rows 256–286 (31 items) have never been used by R0–R2.

They are opened only after all six adapters have completed training.

Required base states on this reserve:
- source 2k = GENERALIZING;
- unadapted target 4k = PATTERN_MATCHING.

No hyperparameter, step count, module, seed set, or selection rule may change
after reserve evaluation begins.

## Ordinary-capability controls

For the target model before adaptation and for every treatment adapter, evaluate:
- 32 ordinary SST-2 sentiment items with correct-label demonstrations;
- 32 conventional arithmetic items without adversarial successive demonstrations.

Both control families must remain confidently GENERALIZING after treatment.
A recovery accompanied by a conventional-control collapse is
`R3_GENERIC_DAMAGE`.

## Success rule

Primary reserve confirmation:
- at least 2 of 3 factual-treatment adapters must classify reserve Truthy as
  `GENERALIZING`;
- 0 of 3 reversed-label controls may classify reserve Truthy as
  `GENERALIZING`;
- all factual-treatment seeds must preserve both ordinary-capability control
  families as `GENERALIZING`;
- exact continuation-token scoring must pass.

Dispositions:
- `R3_PARAMETER_LIGHT_RECOVERY`
- `R3_PARTIAL_SHIFT_NO_RECOVERY`
- `R3_NONSPECIFIC_UPDATE`
- `R3_GENERIC_DAMAGE`
- `NO_R3_RECOVERY`
- `R3_BASELINE_UNRESOLVED`
- `EXACT_BOUNDARY_CONTRACT_FAILED`

## Interpretation

R3 is intrinsically ambiguous between recovery and rapid relearning.

Even `R3_PARAMETER_LIGHT_RECOVERY` does **not** by itself authorize E3
`MECHANISM_PERSISTENCE_SUPPORTED`.

A positive R3 result establishes only that a very small, low-step parameter
intervention can restore the behaviour on fresh examples without conventional
capability collapse. It must be reported as
`PERSISTENCE_UNRESOLVED__RAPID_RELEARNING_NOT_EXCLUDED` unless independent
mechanistic evidence upgrades it.

A negative R3 result also does not establish E4 mechanism change.

Rows 256–286 are consumed by this R3 confirmation and must not be reused as a
fresh holdout afterward.
