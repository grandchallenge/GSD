from gsd.continued_recovery import adjudicate_r4


def base_kwargs():
    return dict(
        source_state="GENERALIZING",
        target_state="PATTERN_MATCHING",
        treatment_recovery_steps=[4, 8, None],
        treatment_control_ok=[True, True, None],
        pattern_generalizing=[False, False, False],
        exact_boundary_ok=True,
    )


def test_r4_success():
    r = adjudicate_r4(**base_kwargs())
    assert r["disposition"] == "R4_CONTINUED_TRAINING_RECOVERY"


def test_r4_partial():
    x = base_kwargs()
    x["treatment_recovery_steps"] = [8, None, None]
    x["treatment_control_ok"] = [True, None, None]
    assert adjudicate_r4(**x)["disposition"] == "R4_PARTIAL_RECOVERY"


def test_r4_no_recovery():
    x = base_kwargs()
    x["treatment_recovery_steps"] = [None, None, None]
    x["treatment_control_ok"] = [None, None, None]
    assert adjudicate_r4(**x)["disposition"] == "NO_R4_RECOVERY"


def test_r4_generic_damage_preempts_success():
    x = base_kwargs()
    x["treatment_control_ok"] = [True, False, None]
    assert adjudicate_r4(**x)["disposition"] == "R4_GENERIC_DAMAGE"


def test_r4_pattern_recovery_is_nonspecific():
    x = base_kwargs()
    x["pattern_generalizing"][0] = True
    assert adjudicate_r4(**x)["disposition"] == "R4_NONSPECIFIC_UPDATE"


def test_r4_baseline_must_resolve():
    x = base_kwargs()
    x["target_state"] = "UNCERTAIN"
    assert adjudicate_r4(**x)["disposition"] == "R4_BASELINE_UNRESOLVED"
