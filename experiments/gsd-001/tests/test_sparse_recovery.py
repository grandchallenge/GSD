from gsd.sparse_recovery import (
    CANDIDATES,
    adjudicate_r2,
    candidate_key,
    choose_sparse_candidate,
)


def test_candidate_order_is_frozen():
    assert CANDIDATES[0] == (12, "attention")
    assert CANDIDATES[1] == (12, "mlp")
    assert CANDIDATES[-1] == (16, "mlp")


def test_choose_first_sparse_recovery():
    states = {candidate_key(layer, kind): "PATTERN_MATCHING" for layer, kind in CANDIDATES}
    states[candidate_key(13, "mlp")] = "GENERALIZING"
    states[candidate_key(15, "attention")] = "GENERALIZING"
    assert choose_sparse_candidate(
        states,
        source_state="GENERALIZING",
        target_state="PATTERN_MATCHING",
        exact_boundary_ok=True,
    ) == "layer13_mlp"


def base_kwargs():
    return dict(
        selected_candidate="layer13_mlp",
        source_heldout_state="GENERALIZING",
        target_heldout_state="PATTERN_MATCHING",
        matched_target_state="GENERALIZING",
        flank_heldout_state="PATTERN_MATCHING",
        matched_flank_state="GENERALIZING",
        identity_state="PATTERN_MATCHING",
        identity_max_margin_drift=0.0,
        shuffled_state="PATTERN_MATCHING",
        control_ok=True,
        exact_boundary_ok=True,
    )


def test_r2_success():
    r = adjudicate_r2(**base_kwargs())
    assert r["disposition"] == "R2_SPARSE_CAUSAL_RECOVERY"
    assert r["evidence_level"] == "E3_MECHANISM_PERSISTENCE_SUPPORTED"


def test_r2_discovery_only_rejected():
    x = base_kwargs()
    x["matched_target_state"] = "UNCERTAIN"
    assert adjudicate_r2(**x)["disposition"] == "R2_DISCOVERY_ONLY_REJECTED"


def test_r2_identity_failure():
    x = base_kwargs()
    x["identity_max_margin_drift"] = 0.01
    assert adjudicate_r2(**x)["disposition"] == "R2_IMPLEMENTATION_CONTROL_FAILED"


def test_r2_shuffle_failure():
    x = base_kwargs()
    x["shuffled_state"] = "GENERALIZING"
    assert adjudicate_r2(**x)["disposition"] == "R2_NONSPECIFIC_MODULE_EFFECT"


def test_r2_no_discovery_recovery():
    x = base_kwargs()
    x["selected_candidate"] = None
    assert adjudicate_r2(**x)["disposition"] == "NO_R2_RECOVERY"
