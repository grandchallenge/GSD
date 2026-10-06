from __future__ import annotations


def adjudicate_r3(
    *,
    source_state: str,
    target_state: str,
    treatment_states: list[str],
    reversed_states: list[str],
    treatment_control_states: list[dict[str, str]],
    exact_boundary_ok: bool,
    base_controls_ok: bool,
) -> dict:
    if not exact_boundary_ok:
        return {"disposition": "EXACT_BOUNDARY_CONTRACT_FAILED"}
    if source_state != "GENERALIZING" or target_state != "PATTERN_MATCHING":
        return {"disposition": "R3_BASELINE_UNRESOLVED"}
    if not base_controls_ok:
        return {"disposition": "R3_BASELINE_UNRESOLVED"}

    for row in treatment_control_states:
        if row.get("sentiment") != "GENERALIZING" or row.get("arithmetic") != "GENERALIZING":
            return {"disposition": "R3_GENERIC_DAMAGE"}

    treatment_generalizing = sum(x == "GENERALIZING" for x in treatment_states)
    reversed_generalizing = sum(x == "GENERALIZING" for x in reversed_states)

    if reversed_generalizing > 0:
        return {
            "disposition": "R3_NONSPECIFIC_UPDATE",
            "treatment_generalizing": treatment_generalizing,
            "reversed_generalizing": reversed_generalizing,
        }

    if treatment_generalizing >= 2:
        return {
            "disposition": "R3_PARAMETER_LIGHT_RECOVERY",
            "treatment_generalizing": treatment_generalizing,
            "reversed_generalizing": 0,
            "interpretation": "PERSISTENCE_UNRESOLVED__RAPID_RELEARNING_NOT_EXCLUDED",
        }

    if any(x in {"GENERALIZING", "UNCERTAIN"} for x in treatment_states):
        return {
            "disposition": "R3_PARTIAL_SHIFT_NO_RECOVERY",
            "treatment_generalizing": treatment_generalizing,
            "reversed_generalizing": 0,
        }

    return {
        "disposition": "NO_R3_RECOVERY",
        "treatment_generalizing": 0,
        "reversed_generalizing": 0,
    }



def adjudicate_r3_control_repair(
    *,
    exact_boundary_ok: bool,
    reproduction_ok: list[bool],
    adapter_control_ok: list[bool],
) -> dict:
    if not exact_boundary_ok:
        return {"disposition": "EXACT_BOUNDARY_CONTRACT_FAILED"}
    if len(reproduction_ok) != 3 or not all(reproduction_ok):
        return {"disposition": "R3_ADAPTER_REPRODUCTION_FAILED"}
    if len(adapter_control_ok) != 3 or not all(adapter_control_ok):
        return {"disposition": "R3_CONTROL_GATE_FAILED"}
    return {
        "disposition": "R3_CONTROL_GATE_REPAIRED",
        "combined_r3_interpretation": (
            "R3_PARAMETER_LIGHT_RECOVERY__PERSISTENCE_UNRESOLVED"
        ),
    }
