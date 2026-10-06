# GSD-WP05U U2B — OLMo3 7B endpoint-control receipt

Disposition: `7B_ENDPOINT_CONTROLS_PASS`

This is a conventional-capability gate only. It does not confirm the behavioural transition.

## Preserved failed execution

Run `20261006T234457Z-23691` from source `ebd4681e1815a3d7f0285cc4ce2bc090b38fae8c` ended `RED_ENGINEERING`.

Cause: the hosted stable-controls dispatcher failed to forward the job's explicit model argument, so the evaluator attempted the frozen 7B revisions against the 1B OLMo2 default repository. The scientific control protocol did not execute.

Repair:
- dispatcher now forwards `--model` for stable controls;
- regression test added;
- frozen model, revisions, control protocol, thresholds, and job were unchanged.

## Fresh exact-head replay

- experiment: `GSD-WP05U-U2B-OLMO3-7B-STABLE-CONTROLS-A100-001`
- run: `20261006T234939Z-26179`
- source commit: `0ad0603045f235bddbeb7a32ce89b40bb25979de`
- hardware: NVIDIA A100-SXM4-40GB
- engineering status: `GREEN_ENGINEERING`
- source payload SHA-256: `c1bd917a60b581ce4c1c6a29fd252523ce64a02d8f1e0d5289a5fe7a63ae23fb`
- job SHA-256: `4cc0ae829dc9b1ae449efca129fdb7b8280d3d0f5465464e449259b1d2e69f75`
- host result SHA-256: `59f0c64efe1ffd69eb769b927646af143bb7cdb371f51b75af141a3403a5b308`

Frozen pair:
`stage1-step565000 -> stage1-step705000`

Paired-control disposition:
- `control_ok: true`
- catastrophic degradation: false
- resolved degradation count: 0/3

Control deltas:
- arithmetic mean-margin delta: +0.30079; CI95 [+0.08914,+0.50566]; hard-accuracy delta 0.0
- language-model mean-token-logprob delta: +0.02442; CI95 [-0.05446,+0.10064]
- sentiment mean-margin delta: +0.34120; CI95 [+0.18090,+0.50672]; hard-accuracy delta -0.0078125

## Adjudication

The frozen 565k→705k candidate is not explained by generic conventional-capability collapse under the existing GSD control contract.

It is therefore authorized to proceed to exact-boundary, multi-seed, prompt-variant adversarial confirmation.

No intermediate checkpoint inside the frozen bracket has been inspected.

Until that confirmation passes, the exact status remains:
`7B_CANDIDATE_TRANSITION`

No mechanistic, cross-transition, E3, or E4 promotion is authorized by U2B.
