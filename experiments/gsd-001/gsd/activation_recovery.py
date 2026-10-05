from __future__ import annotations

PATCH_POINTS = (4, 8, 12, 16)


def choose_discovery_layer(
    discovery_states: dict[int, str],
    *,
    source_state: str,
    target_state: str,
    exact_boundary_ok: bool,
) -> int | None:
    if source_state != "GENERALIZING":
        return None
    if target_state != "PATTERN_MATCHING":
        return None
    if not exact_boundary_ok:
        return None
    for point in PATCH_POINTS:
        if discovery_states.get(point) == "GENERALIZING":
            return point
    return None


def adjudicate_r1(
    *,
    selected_layer: int | None,
    source_heldout_state: str,
    target_heldout_state: str,
    matched_target_state: str | None,
    flank_heldout_state: str,
    matched_flank_state: str | None,
    identity_state: str | None,
    identity_max_margin_drift: float | None,
    shuffled_state: str | None,
    control_ok: bool,
    exact_boundary_ok: bool,
    identity_tolerance: float = 1e-3,
) -> dict:
    if not control_ok:
        return {"disposition": "GENERIC_DEGRADATION_NOT_GSD"}
    if not exact_boundary_ok:
        return {"disposition": "EXACT_BOUNDARY_CONTRACT_FAILED"}
    if selected_layer is None:
        return {"disposition": "NO_R1_RECOVERY"}
    if source_heldout_state != "GENERALIZING" or target_heldout_state != "PATTERN_MATCHING":
        return {"disposition": "R1_HELDOUT_BASELINE_UNRESOLVED"}
    if flank_heldout_state != "PATTERN_MATCHING":
        return {"disposition": "R1_FLANK_BASELINE_UNRESOLVED"}
    if matched_target_state != "GENERALIZING" or matched_flank_state != "GENERALIZING":
        return {"disposition": "R1_DISCOVERY_ONLY_REJECTED"}
    if (
        identity_state != target_heldout_state
        or identity_max_margin_drift is None
        or identity_max_margin_drift > identity_tolerance
    ):
        return {"disposition": "R1_IMPLEMENTATION_CONTROL_FAILED"}
    if shuffled_state == "GENERALIZING":
        return {"disposition": "R1_NONSPECIFIC_ACTIVATION_EFFECT"}
    return {
        "disposition": "R1_ACTIVATION_RECOVERY",
        "evidence_level": "E3_MECHANISM_PERSISTENCE_SUPPORTED",
    }
