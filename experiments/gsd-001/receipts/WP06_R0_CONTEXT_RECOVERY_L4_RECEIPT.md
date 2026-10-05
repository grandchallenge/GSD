# GSD-WP06-R0 — Context-only recovery receipt

Disposition: `R0_INSTRUCTION_ACCESSIBILITY`

Evidence level: `E2_ACCESSIBILITY_SHIFT_SUPPORTED` (bounded context intervention only).

Hosted run:
- experiment: `GSD-WP06-R0-CONTEXT-RECOVERY-L4-001`
- run: `20261005T110356Z-714`
- source commit: `2046a2fe2f8836db64619b1829d99c91c43fd2ad`
- hardware: NVIDIA L4
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 3/3/3
- payload SHA-256: `ecd5ac46e063a3a0b3283733ef09ed907fd9afe776c8aaf0ae12e26aaa429096`
- job SHA-256: `363beb950df945dacaf5ea781d32f2a7a40169848d1c8da8359b729e2bafe15c`
- result SHA-256: `0a86485e2944cea8fbc0abbcb00f11ca898433a8f8b729f44a76e56b08537830`
- protocol: `experiments/gsd-001/WP06_R0_PROTOCOL.md`

Confirmed transition under test:
- task: `truthy_answer/surprising_truth`
- source: step 2000 = GENERALIZING
- flank: step 3000 = PATTERN_MATCHING under base context
- target: step 4000 = PATTERN_MATCHING under base context
- inherited broad stable-control pair 2k -> 4k: pass
- exact answer-boundary gate: pass

## Source step 2000

Every frozen context variant remains GENERALIZING:
- base_k8
- reverse_k8
- sorted_k8
- k1
- k4
- k16
- truth_instruction_k8
- zero_shot_truth

## Target step 4000

Remains PATTERN_MATCHING under:
- base_k8
- reverse_k8
- sorted_k8
- k1
- k4
- k16
- truth_instruction_k8

Recovers to GENERALIZING only under:
- `zero_shot_truth`

No structural context variant recovered the target.

## Flank step 3000

The same pattern holds:
- all demo-bearing variants remain PATTERN_MATCHING;
- `zero_shot_truth` is GENERALIZING.

## Interpretation

The confirmed 2k -> 4k behavioural loss is not a broad loss of the ability to answer the Truthy items according to factual truth. A bounded context-only intervention—removing the demonstrations and giving the factual-truth instruction—restores GENERALIZING behaviour at both the 3k flank and 4k target.

This supports an accessibility/expression shift at evidence level E2.

It does **not** establish:
- literal persistence of the same internal mechanism;
- a specific circuit or representation surviving;
- capacity-allocation competition;
- optimizer causation;
- architecture causation.

The fact that `truth_instruction_k8` does not recover while `zero_shot_truth` does is material: the demonstrations themselves remain sufficient to drive pattern-matching even when the instruction explicitly requests factual truth.

Next WP06 tier: R1 activation-space recovery, to test whether a bounded internal intervention can restore the generalizing state under the original demo-bearing context.
