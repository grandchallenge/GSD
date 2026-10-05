from gsd.activation_recovery import (
    PATCH_POINTS,
    adjudicate_r1,
    choose_discovery_layer,
)


def test_choose_earliest_discovery_layer():
    states = {4: "PATTERN_MATCHING", 8: "GENERALIZING", 12: "GENERALIZING", 16: "GENERALIZING"}
    assert choose_discovery_layer(
        states,
        source_state="GENERALIZING",
        target_state="PATTERN_MATCHING",
        exact_boundary_ok=True,
    ) == 8


def test_no_layer_if_baseline_invalid():
    states = {p: "GENERALIZING" for p in PATCH_POINTS}
    assert choose_discovery_layer(
        states,
        source_state="UNCERTAIN",
        target_state="PATTERN_MATCHING",
        exact_boundary_ok=True,
    ) is None


def kwargs():
    return dict(
        selected_layer=8,
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


def test_r1_success():
    r = adjudicate_r1(**kwargs())
    assert r["disposition"] == "R1_ACTIVATION_RECOVERY"
    assert r["evidence_level"] == "E3_MECHANISM_PERSISTENCE_SUPPORTED"


def test_identity_failure():
    x = kwargs()
    x["identity_max_margin_drift"] = 0.01
    assert adjudicate_r1(**x)["disposition"] == "R1_IMPLEMENTATION_CONTROL_FAILED"


def test_shuffle_recovery_is_nonspecific():
    x = kwargs()
    x["shuffled_state"] = "GENERALIZING"
    assert adjudicate_r1(**x)["disposition"] == "R1_NONSPECIFIC_ACTIVATION_EFFECT"


def test_heldout_failure_rejects_discovery():
    x = kwargs()
    x["matched_target_state"] = "PATTERN_MATCHING"
    assert adjudicate_r1(**x)["disposition"] == "R1_DISCOVERY_ONLY_REJECTED"


def test_no_discovery_recovery():
    x = kwargs()
    x["selected_layer"] = None
    assert adjudicate_r1(**x)["disposition"] == "NO_R1_RECOVERY"
