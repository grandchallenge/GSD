# GSD-WP01 — Accelerator Sweep Handoff

GCL-ZERO-CONTEXT-LAUNCH/2

STATE: ACTIVE  
CAMPAIGN: GSD-001  
TRANCHE: GSD-TRANCHE-A  
ASSIGNMENT_ID: GSD-WP01-ACCELERATOR  
INTENDED_RETURN: grandchallenge/GSD issue #1  
RETURN_PROTOCOL: GCL-CONTRIBUTION-RESULT/1  
CANONICAL_MUTATION_AUTHORIZED: NO  
CERTIFICATION_AUTHORIZED: NO  
EXECUTION_AUTHORIZED: YES

## Mission

Execute the first real checkpoint sweep for Generalization-State Dynamics.

Do not infer a mechanism. Do not call a state reversal a candidate transition until
stable-control evidence exists.

## Protected work surface

Use branch:

main

Read first:

1. experiments/gsd-001/CAMPAIGN_STATE.yaml
2. experiments/gsd-001/source-lock.yaml
3. experiments/gsd-001/receipts/WP00_RECEIPT.md
4. experiments/gsd-001/README.md

The evaluator infrastructure has already passed its dedicated CI and the repository validator.

## External source identity

The behavioural runner must be:

Jiaxin-Wen/GDsuite
commit e3f5de327cd90a579e50a89f04fa167c584b799f

The GCL wrapper verifies the load-bearing Git blobs and fails closed on mismatch.

Primary model series:

allenai/OLMo-2-0425-1B-early-training

Primary first-pass task:

intuitive_answer/crt

## Runtime

CUDA is required.

Preferred: L4, A100, or later NVIDIA GPU.

T4 is allowed for discovery. Because T4 lacks native BF16, the GCL runner will use its
declared FP16 config override. Any apparent transition found only under FP16 must later
be replayed under BF16 before promotion.

## Execution

First perform the one-checkpoint smoke test documented in README.md.

The smoke test must produce:

- upstream GDsuite output;
- items.jsonl;
- summary.json;
- manifest.json with exact model and dataset SHAs.

If the smoke test is valid, execute the coarse sweep:

- automatic checkpoint enumeration;
- revision stride 2;
- final checkpoint forcibly included;
- expected checkpoint count: 20;
- family: intuitive_answer;
- n_seeds: 1;
- max_eval: 128.

Do not silently reduce the checkpoint count below 20.

Resource tuning may reduce vLLM concurrency. Record any runtime override.

## Analysis

After all 20 checkpoint summaries exist, run the task-level catalogue for:

intuitive_answer/crt

Without stable-control evidence, any confident sign/state flip must remain:

STATE_REVERSAL_REQUIRES_CONTROL_VALIDATION

It must not be returned as CANDIDATE_TRANSITION.

## Required return

Return one durable result with:

- GPU model and VRAM;
- Python, CUDA, PyTorch, Transformers, vLLM versions;
- exact dataset SHA;
- all 20 model revision names and resolved SHAs;
- precision mode for each run;
- count of successful and failed revisions;
- compact table of task-level mean margin, CI, and state by checkpoint;
- state-reversal catalogue;
- artifact location for manifests, summaries, and items;
- any OOM/runtime repairs;
- strongest supported disposition.

Allowed dispositions:

- WP01_SWEEP_COMPLETE_NO_STATE_REVERSAL
- WP01_SWEEP_COMPLETE_STATE_REVERSAL_FOUND
- WP01_PARTIAL_RUNTIME_BLOCKER
- WP01_SOURCE_IDENTITY_BLOCKER

A no-reversal result on this task does not imply absence of generalization-state dynamics.
It triggers the next predeclared broader-probe step.

## Prohibitions

Do not:

- alter the paper/source lock;
- change answer semantics;
- search checkpoints post hoc for a prettier transition and omit the rest;
- promote hard accuracy without the soft margin;
- claim mechanism persistence/loss;
- launch 7B or 32B work;
- modify GSD canonical main as part of this assignment.
