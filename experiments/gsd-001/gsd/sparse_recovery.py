from __future__ import annotations

CANDIDATES = tuple(
    (layer, kind)
    for layer in range(12, 17)
    for kind in ("attention", "mlp")
)


def candidate_key(layer: int, kind: str) -> str:
    return f"layer{layer}_{kind}"


def choose_sparse_candidate(
    discovery_states: dict[str, str],
    *,
    source_state: str,
    target_state: str,
    exact_boundary_ok: bool,
) -> str | None:
    if source_state != "GENERALIZING":
        return None
    if target_state != "PATTERN_MATCHING":
        return None
    if not exact_boundary_ok:
        return None
    for layer, kind in CANDIDATES:
        key = candidate_key(layer, kind)
        if discovery_states.get(key) == "GENERALIZING":
            return key
    return None


def adjudicate_r2(
    *,
    selected_candidate: str | None,
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
    if selected_candidate is None:
        return {"disposition": "NO_R2_RECOVERY"}
    if source_heldout_state != "GENERALIZING" or target_heldout_state != "PATTERN_MATCHING":
        return {"disposition": "R2_HELDOUT_BASELINE_UNRESOLVED"}
    if flank_heldout_state != "PATTERN_MATCHING":
        return {"disposition": "R2_FLANK_BASELINE_UNRESOLVED"}
    if matched_target_state != "GENERALIZING" or matched_flank_state != "GENERALIZING":
        return {"disposition": "R2_DISCOVERY_ONLY_REJECTED"}
    if (
        identity_state != target_heldout_state
        or identity_max_margin_drift is None
        or identity_max_margin_drift > identity_tolerance
    ):
        return {"disposition": "R2_IMPLEMENTATION_CONTROL_FAILED"}
    if shuffled_state == "GENERALIZING":
        return {"disposition": "R2_NONSPECIFIC_MODULE_EFFECT"}
    return {
        "disposition": "R2_SPARSE_CAUSAL_RECOVERY",
        "evidence_level": "E3_MECHANISM_PERSISTENCE_SUPPORTED",
    }
