# GSD-WP02 — Truthy stable-control validation receipt

Disposition: `CONTROL_VALIDATED__THREE_CANDIDATE_TRANSITIONS`

Hosted run:
- experiment: `GSD-WP02-TRUTHY-STABLE-CONTROLS-L4-001`
- source commit: `dc1d895623e4b99801c80b2b4f1f2c65d81a3129`
- hardware: NVIDIA L4
- status: `GREEN_ENGINEERING`
- checkpoints: 3/3
- summaries: 3/3
- manifests: 3/3
- source payload SHA-256: `b425a7f2386b11cdbe322c893af320899108b63f80d6ece280c9dac18d854a0b`
- job SHA-256: `3a1d432d2f541af68c1051befa6e3a875e2ceba6936f8596215dd5e688d8c52a`
- result SHA-256: `deb68c84d77a16f199c625ef4bc476ff0990ddeb9329cbed91bc128e9d266385`

Controls:
- ordinary SST-2 correct-label ICL;
- zero-shot arithmetic;
- held-out SST-2 review token log-likelihood.

Pair gate:
- 0 -> 2000: `control_ok: true`, zero resolved degradations, no catastrophic degradation.
- 2000 -> 4000: `control_ok: true`, zero resolved degradations, no catastrophic degradation.

Key deltas:
- 0 -> 2000: arithmetic hard accuracy +0.1641; LM mean token log-prob +4.3936; sentiment hard accuracy -0.0625 with unresolved margin change.
- 2000 -> 4000: arithmetic hard accuracy +0.1563; LM mean token log-prob +1.1545; sentiment hard accuracy +0.2500.

Controlled transition catalogue:
- `truthy_answer/surprising_truth`: 0 PATTERN_MATCHING -> 2000 GENERALIZING: `CANDIDATE_TRANSITION`.
- `truthy_answer/surprising_truth`: 2000 GENERALIZING -> 4000 PATTERN_MATCHING: `CANDIDATE_TRANSITION`.
- `truthy_answer/common_misconception`: 0 GENERALIZING -> 2000 PATTERN_MATCHING: `CANDIDATE_TRANSITION`.

These are behavioural candidate transitions only. WP03 adversarial validation is required before any `CONFIRMED_TRANSITION` disposition.
