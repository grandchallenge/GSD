# GSD-WP06-R0 — Context-only recovery

Status: frozen before execution.

Parent work package: `GSD-WP06 — Recovery and Residual test`.

Confirmed transition under test:
- task: `truthy_answer/surprising_truth`
- source: `stage1-step2000-tokens5B` = GENERALIZING
- primary target: `stage1-step4000-tokens9B` = PATTERN_MATCHING
- localization flank: `stage1-step3000-tokens7B` = PATTERN_MATCHING
- stable-control pair 2k -> 4k: pass

Purpose:
test the smallest intervention class, R0, before any activation or parameter intervention.

Exact evaluation contract:
- dataset: `jiaxin-wen/generalization-dynamics-evals`
- config: `truthy_answer.surprising_truth`
- first 128 test rows
- sum log-probability margin is primary
- exact continuation boundary contract from WP03R
- seed 0 demonstration sampling where sampling is used
- BF16 on L4

Frozen context variants:
1. `base_k8` — reference only; not a recovery intervention.
2. `reverse_k8` — same seed-0 eight demonstrations, reversed.
3. `sorted_k8` — same seed-0 eight demonstrations, sorted lexically by prompt.
4. `k1` — one demonstration.
5. `k4` — four demonstrations.
6. `k16` — sixteen demonstrations.
7. `truth_instruction_k8` — eight demonstrations plus explicit instruction to answer according to factual truth rather than superficial example patterns.
8. `zero_shot_truth` — no demonstrations; same factual-truth instruction.

Recovery gate for a variant:
- source step 2k under that variant must remain `GENERALIZING`;
- primary target step 4k under that variant must be `GENERALIZING`;
- exact answer-boundary gate must pass;
- inherited broad stable controls for 2k -> 4k must remain true.

Dispositions:
- if any of `reverse_k8`, `sorted_k8`, `k1`, `k4`, or `k16` recovers: `R0_CONTEXT_RECOVERY`;
- otherwise if only `truth_instruction_k8` or `zero_shot_truth` recovers: `R0_INSTRUCTION_ACCESSIBILITY`;
- otherwise: `NO_R0_RECOVERY`.

Interpretation:
- R0 recovery may support at most E2 `ACCESSIBILITY_SHIFT_SUPPORTED`;
- R0 failure does not prove mechanism loss; it advances the programme to R1 activation-space recovery;
- no mechanism-persistence or mechanism-change claim is authorized from R0 alone.

Step 3k is supplementary. Recovery there cannot substitute for the primary 4k gate.
