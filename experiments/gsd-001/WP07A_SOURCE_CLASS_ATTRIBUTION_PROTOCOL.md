# GSD-WP07A — source-class micro-continuation attribution pilot

Status: frozen before execution.

Parent: `GSD-WP07 — Data-window attribution`.

## Question

Which published OLMo Stage-1 source classes can move the confirmed 1B surprising-truth state in the same direction as the 2k→3k transition?

This is an attribution-discovery pilot. It does not by itself satisfy the WP07 exit gate.

## Source lock

Confirmed behavioural transition:
- model: `allenai/OLMo-2-0425-1B-early-training`;
- source: `stage1-step2000-tokens5B` = GENERALIZING;
- flank: `stage1-step3000-tokens7B` = PATTERN_MATCHING;
- task: `truthy_answer/surprising_truth`.

Published Stage-1 recipe source:
- repository: `allenai/OLMo`;
- source commit: `090253dac6688f2532509daa7aa2eb5fae50e956`;
- config: `configs/official-0425/OLMo2-1B-stage1.yaml`;
- provenance: `configs/official-1124/provenance.csv`;
- dataset: `allenai/olmo-mix-1124`.

Published source classes:
1. `algebraic-stack`
2. `arxiv`
3. `dclm`
4. `open-web-math`
5. `pes2o`
6. `starcoder`
7. `wiki`

The published loader deterministically shuffles the concatenated example index using PCG64 with the training seed. The early-training release documents the same architecture and starting checkpoint as the official 1B run, but does not establish exact identity of the realized per-example training order. Therefore this tranche makes **no exact original-window reconstruction claim**.

## Pilot intervention

Start independently from the frozen 2k source checkpoint for every source class.

For each class:
- stream text from the matching `allenai/olmo-mix-1124` configuration;
- deterministic document shuffle seed: 1729;
- tokenize with the source checkpoint tokenizer;
- sequence length: 4096;
- microbatch: 2 sequences;
- gradient accumulation: 4;
- optimizer steps: 8;
- effective training tokens: 262,144 per class;
- optimizer: AdamW;
- learning rate: 2.4e-4;
- betas: (0.9, 0.95);
- eps: 1e-8;
- weight decay: 0.1;
- gradient clipping: 1.0;
- BF16 on A100.

The learning rate is a controlled early-stage micro-continuation value chosen near the published Stage-1 warmup scale. It is not claimed to reconstruct the exact original optimizer state, which is unavailable.

## Discovery evaluation

Use surprising-truth rows 0–63, already programme-consumed, only for attribution discovery.

For each class, measure:
1. paired change in mean behavioural margin from the untouched 2k source;
2. post-continuation state classification;
3. cosine alignment of layer-13 residual movement with the original 2k→3k layer-13 transition direction;
4. cosine alignment of Q/layer-14/head-2 movement with the original 2k→3k operator direction;
5. training loss trajectory.

Internal statistics are measured at the final prompt token before candidate answers, matching WP05.

## Candidate selection

Define `toward_transition_score = - behavioural_margin_delta`.

Eligible candidates must have:
- `toward_transition_score > 0`;
- layer-13 transition-direction cosine > 0;
- Q14/H2 transition-direction cosine > 0.

If at least one class is eligible, select exactly one candidate:
1. highest `toward_transition_score`;
2. tie-break by Q14/H2 alignment;
3. second tie-break by layer-13 alignment.

No post-hoc alternative class may be substituted after selection.

If no class is eligible, disposition is:
`WP07A_NO_CLASS_ATTRIBUTION_CANDIDATE`.

If one is selected:
`WP07A_CLASS_ATTRIBUTION_CANDIDATE`.

## Claim boundary

A WP07A candidate is exploratory attribution evidence only.

It does not establish:
- that the class occurred disproportionately in the original 2k→3k window;
- causal identity of the original transition;
- held-out attribution;
- WP07 completion;
- E3/E4 promotion.

A selected class must be tested in a separately frozen WP07B held-out micro-continuation against matched controls before any attribution hypothesis survives the programme exit gate.

## Coupling to WP05 utility expansion

WP07A deliberately treats the frozen WP05 layer-13 and Q14/H2 signatures as dependent variables, not assumed causes.

A source class is more useful if its behavioural movement is accompanied by internal movement in the already-frozen transition directions.
