from gsd.recovery import adjudicate_r0


def base_source():
    return {
        "reverse_k8": "GENERALIZING",
        "sorted_k8": "GENERALIZING",
        "k1": "GENERALIZING",
        "k4": "GENERALIZING",
        "k16": "GENERALIZING",
        "truth_instruction_k8": "GENERALIZING",
        "zero_shot_truth": "GENERALIZING",
    }


def test_structural_context_recovery():
    target = {k: "PATTERN_MATCHING" for k in base_source()}
    target["k4"] = "GENERALIZING"
    r = adjudicate_r0(base_source(), target, control_ok=True, exact_boundary_ok=True)
    assert r["disposition"] == "R0_CONTEXT_RECOVERY"
    assert r["structural_recoveries"] == ["k4"]


def test_instruction_only_recovery():
    target = {k: "PATTERN_MATCHING" for k in base_source()}
    target["truth_instruction_k8"] = "GENERALIZING"
    r = adjudicate_r0(base_source(), target, control_ok=True, exact_boundary_ok=True)
    assert r["disposition"] == "R0_INSTRUCTION_ACCESSIBILITY"
    assert r["instruction_recoveries"] == ["truth_instruction_k8"]


def test_no_r0_recovery():
    target = {k: "PATTERN_MATCHING" for k in base_source()}
    r = adjudicate_r0(base_source(), target, control_ok=True, exact_boundary_ok=True)
    assert r["disposition"] == "NO_R0_RECOVERY"


def test_control_failure_preempts_recovery():
    target = {k: "GENERALIZING" for k in base_source()}
    r = adjudicate_r0(base_source(), target, control_ok=False, exact_boundary_ok=True)
    assert r["disposition"] == "GENERIC_DEGRADATION_NOT_GSD"


def test_boundary_failure_preempts_recovery():
    target = {k: "GENERALIZING" for k in base_source()}
    r = adjudicate_r0(base_source(), target, control_ok=True, exact_boundary_ok=False)
    assert r["disposition"] == "EXACT_BOUNDARY_CONTRACT_FAILED"


def test_source_must_remain_generalizing():
    source = base_source()
    source["k1"] = "UNCERTAIN"
    target = {k: "PATTERN_MATCHING" for k in base_source()}
    target["k1"] = "GENERALIZING"
    r = adjudicate_r0(source, target, control_ok=True, exact_boundary_ok=True)
    assert r["disposition"] == "NO_R0_RECOVERY"
    assert "k1" in r["invalid_source_variants"]
