# GSD-WP05U U2B — OLMo3 7B candidate-control protocol

Status: frozen before execution.

Parent discovery receipt:
`experiments/gsd-001/receipts/WP05U_U2A_OLMO3_7B_COARSE_A100_RECEIPT.md`

Frozen candidate:
- model: `allenai/Olmo-3-1025-7B`
- left endpoint: `stage1-step565000` = PATTERN_MATCHING on coarse surprising-truth scan;
- right endpoint: `stage1-step705000` = GENERALIZING;
- no checkpoint inside this interval may be inspected before endpoint confirmation is adjudicated.

Question: does the candidate coincide with generic conventional-capability collapse?

Run the existing GSD stable-control evaluator at both endpoints with the same model and dataset identity.

Control gate:
- the endpoint pair must receive `control_ok: true` under the existing paired-control disposition;
- failure disposition is `GENERIC_DEGRADATION_NOT_GSD`;
- success only permits exact-boundary adversarial replay. It does not confirm the transition.

Claim boundary:
`7B endpoint conventional-control validation only; no transition or mechanistic claim`.
