# GSD-WP06-R3C — Conventional-control gate repair receipt

Disposition: `R3_CONTROL_GATE_REPAIRED`

Combined R3 interpretation:
`R3_PARAMETER_LIGHT_RECOVERY__PERSISTENCE_UNRESOLVED`

Hosted run:
- experiment: `GSD-WP06-R3C-CONTROL-GATE-REPAIR-L4-001`
- run: `20261006T075704Z-1966`
- source commit: `7ea44b1b0a2c01f8ee6995d5a97037b4b9be5300`
- hardware: NVIDIA L4
- engineering status: `GREEN_ENGINEERING`
- revisions/summaries/manifests: 1/1/1
- payload SHA-256: `fcd9d9b4be567a081accc5d568d3655547043d9f25ffdcc2c5770cdb5e324925`
- job SHA-256: `6fabead011594d3cf5efdd00cb55d974af05bb30873c2b28925c358600be7376`
- result SHA-256: `178b1d161932896c48ce77b3087cf086ce81d0b7e9984e26d7fb53e73f882c55`
- protocol: `experiments/gsd-001/WP06_R3_CONTROL_REPAIR_PROTOCOL.md`

## Scope

R3C repaired only the ordinary-capability control gate.

It did **not**:
- rescore Truthy reserve rows 256-286;
- change the R3 adapter module, rank, training data, optimizer, step budget, or seeds;
- alter the primary R3 recovery criterion;
- promote an E3 persistence claim.

## Adapter reproduction

The three treatment adapters were independently reproduced with the exact R3 recipe.

Seed 0:
- delta Frobenius norm: 4.369599819
- original R3 norm: 4.369599819
- reproduction check: pass
- final pairwise loss: 1.82e-4

Seed 1:
- delta Frobenius norm: 4.484992981
- original R3 norm: 4.485442162
- reproduction check: pass
- final pairwise loss: 1.35e-4

Seed 2:
- delta Frobenius norm: 4.415066719
- original R3 norm: 4.415067673
- reproduction check: pass
- final pairwise loss: 2.03e-4

Thus 3/3 adapter reproductions satisfy the preregistered norm/loss sanity checks.

## 128-item conventional control repair

The unadapted 4k baseline was compared with each reproduced treatment adapter on:
- 128 ordinary SST-2 sentiment items;
- 128 conventional arithmetic items;
- 128 held-out review language-model texts.

The repair criterion was strict:
zero of the three families may show a resolved negative paired shift.

Seed 0:
- sentiment mean-margin delta: +0.0657
- arithmetic mean-margin delta: +0.2592
- LM mean token-logprob delta: -0.00009
- resolved degradations: 0
- `control_ok: true`

Seed 1:
- sentiment mean-margin delta: +0.0725
- arithmetic mean-margin delta: +0.2620
- LM mean token-logprob delta: +0.00035
- resolved degradations: 0
- `control_ok: true`

Seed 2:
- sentiment mean-margin delta: +0.0703
- arithmetic mean-margin delta: +0.2420
- LM mean token-logprob delta: +0.00091
- resolved degradations: 0
- `control_ok: true`

Thus 3/3 independently reproduced treatment adapters preserve the conventional capability suite under the repaired paired-degradation gate.

## Combined R3 conclusion

The immutable primary R3 run established:
- untouched Truthy reserve reproduces the 2k GENERALIZING / 4k PATTERN_MATCHING distinction;
- 3/3 rank-1 factual-treatment adapters recover GENERALIZING on the reserve;
- 0/3 matched reversed-label adapters recover; all push strongly PATTERN_MATCHING;
- only 4096 trainable parameters;
- one 64-example pass and 32 optimizer steps.

R3C establishes that independently reproduced treatment adapters do not cause resolved degradation on the full conventional control suite.

The combined, defensible disposition is therefore:

`R3_PARAMETER_LIGHT_RECOVERY__PERSISTENCE_UNRESOLVED`

This means a very small and short parameter intervention can robustly restore the behaviour without detected ordinary-capability collapse.

It does **not** distinguish:
- recovery of a latent pre-existing mechanism; from
- rapid task-specific relearning.

Therefore:
- no E3 `MECHANISM_PERSISTENCE_SUPPORTED` promotion;
- no E4 mechanism-change claim;
- strongest programme-level evidence remains `E2_ACCESSIBILITY_SHIFT_SUPPORTED`.

## Next recovery tier

R4 — continued-training recovery.

Any R4 design must explicitly treat all 287 `surprising_truth` test rows as previously consumed and must not describe any subset as a fresh holdout.
