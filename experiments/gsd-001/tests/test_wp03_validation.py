from gsd.wp03_validation import (
    CandidateSpec,
    CONFIRMED,
    CONTROL_FAIL,
    PROMPT_RISK,
    REPLAY_FAIL,
    TOKEN_RISK,
    validate_candidate,
)


SPEC = CandidateSpec(
    task="truthy_answer/surprising_truth",
    from_revision="r0",
    to_revision="r2",
    from_state="PATTERN_MATCHING",
    to_state="GENERALIZING",
)


def _variants(states=("PATTERN_MATCHING", "GENERALIZING")):
    return {
        "instruction_prefix": states,
        "expanded_markers": states,
        "double_newline": states,
    }


def _replay(good=4):
    out = {}
    for seed in range(4):
        out[seed] = (
            ("PATTERN_MATCHING", "GENERALIZING")
            if seed < good
            else ("UNCERTAIN", "GENERALIZING")
        )
    return out


def test_confirmed_candidate():
    out = validate_candidate(
        SPEC,
        control_ok=True,
        endpoint_boundaries_stable=True,
        variant_endpoint_states=_variants(),
        replay_endpoint_states=_replay(3),
    )
    assert out["disposition"] == CONFIRMED
    assert out["replay_support"] == 3


def test_control_failure_preempts_other_gates():
    out = validate_candidate(
        SPEC,
        control_ok=False,
        endpoint_boundaries_stable=True,
        variant_endpoint_states=_variants(),
        replay_endpoint_states=_replay(4),
    )
    assert out["disposition"] == CONTROL_FAIL


def test_token_boundary_failure():
    out = validate_candidate(
        SPEC,
        control_ok=True,
        endpoint_boundaries_stable=False,
        variant_endpoint_states=_variants(),
        replay_endpoint_states=_replay(4),
    )
    assert out["disposition"] == TOKEN_RISK


def test_prompt_failure():
    variants = _variants()
    variants["double_newline"] = ("UNCERTAIN", "GENERALIZING")
    out = validate_candidate(
        SPEC,
        control_ok=True,
        endpoint_boundaries_stable=True,
        variant_endpoint_states=variants,
        replay_endpoint_states=_replay(4),
    )
    assert out["disposition"] == PROMPT_RISK


def test_independent_replay_failure():
    out = validate_candidate(
        SPEC,
        control_ok=True,
        endpoint_boundaries_stable=True,
        variant_endpoint_states=_variants(),
        replay_endpoint_states=_replay(2),
    )
    assert out["disposition"] == REPLAY_FAIL
