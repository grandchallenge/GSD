from __future__ import annotations


def adjudicate_r4(
    *,
    source_state: str,
    target_state: str,
    treatment_recovery_steps: list[int | None],
    treatment_control_ok: list[bool | None],
    pattern_generalizing: list[bool],
    exact_boundary_ok: bool,
) -> dict:
    if not exact_boundary_ok:
        return {"disposition": "EXACT_BOUNDARY_CONTRACT_FAILED"}
    if source_state != "GENERALIZING" or target_state != "PATTERN_MATCHING":
        return {"disposition": "R4_BASELINE_UNRESOLVED"}
    if len(treatment_recovery_steps) != 3 or len(treatment_control_ok) != 3:
        raise ValueError("R4 requires exactly three treatment seeds")
    if len(pattern_generalizing) != 3:
        raise ValueError("R4 requires exactly three pattern-control seeds")

    if any(pattern_generalizing):
        return {
            "disposition": "R4_NONSPECIFIC_UPDATE",
            "recovered_treatment_seeds": sum(x is not None for x in treatment_recovery_steps),
            "pattern_generalizing_seeds": sum(pattern_generalizing),
        }

    recovered = [i for i, step in enumerate(treatment_recovery_steps) if step is not None]
    for i in recovered:
        if treatment_control_ok[i] is not True:
            return {
                "disposition": "R4_GENERIC_DAMAGE",
                "recovered_treatment_seeds": len(recovered),
                "failed_control_seed": i,
            }

    if len(recovered) >= 2:
        return {
            "disposition": "R4_CONTINUED_TRAINING_RECOVERY",
            "recovered_treatment_seeds": len(recovered),
            "recovery_steps": treatment_recovery_steps,
        }
    if len(recovered) == 1:
        return {
            "disposition": "R4_PARTIAL_RECOVERY",
            "recovered_treatment_seeds": 1,
            "recovery_steps": treatment_recovery_steps,
        }
    return {
        "disposition": "NO_R4_RECOVERY",
        "recovered_treatment_seeds": 0,
        "recovery_steps": treatment_recovery_steps,
    }
