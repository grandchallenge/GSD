from __future__ import annotations

STRUCTURAL_CONTEXT_VARIANTS = (
    "reverse_k8",
    "sorted_k8",
    "k1",
    "k4",
    "k16",
)
INSTRUCTION_VARIANTS = (
    "truth_instruction_k8",
    "zero_shot_truth",
)


def adjudicate_r0(
    source_states: dict[str, str],
    target_states: dict[str, str],
    *,
    control_ok: bool,
    exact_boundary_ok: bool,
) -> dict:
    structural = []
    instruction = []
    invalid_source = []

    if not control_ok:
        return {
            "disposition": "GENERIC_DEGRADATION_NOT_GSD",
            "structural_recoveries": [],
            "instruction_recoveries": [],
            "invalid_source_variants": [],
        }
    if not exact_boundary_ok:
        return {
            "disposition": "EXACT_BOUNDARY_CONTRACT_FAILED",
            "structural_recoveries": [],
            "instruction_recoveries": [],
            "invalid_source_variants": [],
        }

    for variant in (*STRUCTURAL_CONTEXT_VARIANTS, *INSTRUCTION_VARIANTS):
        source = source_states.get(variant)
        target = target_states.get(variant)
        if source != "GENERALIZING":
            invalid_source.append(variant)
            continue
        if target != "GENERALIZING":
            continue
        if variant in STRUCTURAL_CONTEXT_VARIANTS:
            structural.append(variant)
        else:
            instruction.append(variant)

    if structural:
        disposition = "R0_CONTEXT_RECOVERY"
    elif instruction:
        disposition = "R0_INSTRUCTION_ACCESSIBILITY"
    else:
        disposition = "NO_R0_RECOVERY"

    return {
        "disposition": disposition,
        "structural_recoveries": structural,
        "instruction_recoveries": instruction,
        "invalid_source_variants": invalid_source,
    }
