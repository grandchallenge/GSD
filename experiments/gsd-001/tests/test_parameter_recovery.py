from gsd.parameter_recovery import adjudicate_r3


def controls():
    return [
        {"sentiment": "GENERALIZING", "arithmetic": "GENERALIZING"},
        {"sentiment": "GENERALIZING", "arithmetic": "GENERALIZING"},
        {"sentiment": "GENERALIZING", "arithmetic": "GENERALIZING"},
    ]


def base_kwargs():
    return dict(
        source_state="GENERALIZING",
        target_state="PATTERN_MATCHING",
        treatment_states=["GENERALIZING", "GENERALIZING", "UNCERTAIN"],
        reversed_states=["PATTERN_MATCHING", "PATTERN_MATCHING", "UNCERTAIN"],
        treatment_control_states=controls(),
        exact_boundary_ok=True,
        base_controls_ok=True,
    )


def test_r3_success_requires_two_of_three_and_no_reverse_recovery():
    r = adjudicate_r3(**base_kwargs())
    assert r["disposition"] == "R3_PARAMETER_LIGHT_RECOVERY"
    assert r["treatment_generalizing"] == 2


def test_r3_reversed_recovery_rejects_specificity():
    x = base_kwargs()
    x["reversed_states"][1] = "GENERALIZING"
    assert adjudicate_r3(**x)["disposition"] == "R3_NONSPECIFIC_UPDATE"


def test_r3_control_damage_preempts_recovery():
    x = base_kwargs()
    x["treatment_control_states"][0]["sentiment"] = "PATTERN_MATCHING"
    assert adjudicate_r3(**x)["disposition"] == "R3_GENERIC_DAMAGE"


def test_r3_partial_shift():
    x = base_kwargs()
    x["treatment_states"] = ["UNCERTAIN", "PATTERN_MATCHING", "PATTERN_MATCHING"]
    assert adjudicate_r3(**x)["disposition"] == "R3_PARTIAL_SHIFT_NO_RECOVERY"


def test_r3_no_recovery():
    x = base_kwargs()
    x["treatment_states"] = ["PATTERN_MATCHING"] * 3
    assert adjudicate_r3(**x)["disposition"] == "NO_R3_RECOVERY"


def test_r3_baseline_must_resolve():
    x = base_kwargs()
    x["target_state"] = "UNCERTAIN"
    assert adjudicate_r3(**x)["disposition"] == "R3_BASELINE_UNRESOLVED"
