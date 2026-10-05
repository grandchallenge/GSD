from gsd.scoring import STATE_GENERALIZING, STATE_PATTERN
from gsd.transitions import (
    CheckpointSummary,
    adversarial_disposition,
    detect_candidate_transitions,
    detect_state_reversals,
)


def _summary(step, state, margin, control_ok=True):
    return CheckpointSummary(
        revision=f"stage1-step{step}",
        step=step,
        state=state,
        mean_margin=margin,
        ci_low=margin - 0.1,
        ci_high=margin + 0.1,
        n=100,
        control_ok=control_ok,
    )


def test_candidate_transition_detected():
    rows = [
        _summary(1000, STATE_GENERALIZING, 0.8),
        _summary(2000, STATE_PATTERN, -0.7),
    ]
    out = detect_candidate_transitions(rows)
    assert len(out) == 1
    assert out[0]["status"] == "CANDIDATE_TRANSITION"


def test_control_failure_not_promoted():
    rows = [
        _summary(1000, STATE_GENERALIZING, 0.8),
        _summary(2000, STATE_PATTERN, -0.7, control_ok=False),
    ]
    assert detect_candidate_transitions(rows) == []


def test_adversarial_disposition():
    assert adversarial_disposition(
        base_margin=1.0,
        prompt_variant_margins=[0.2, 0.5, 0.1, 0.8],
        token_boundary_stable=True,
        soft_margin_resolved=True,
        control_ok=True,
    ) == "SURVIVES_ADVERSARIAL_CHECKS"
    assert adversarial_disposition(
        base_margin=1.0,
        prompt_variant_margins=[0.2, -0.5],
        token_boundary_stable=True,
        soft_margin_resolved=True,
        control_ok=True,
    ) == "PROMPT_SENSITIVE_UNCONFIRMED"


def test_reversal_without_control_is_not_candidate():
    rows = [
        _summary(1000, STATE_GENERALIZING, 0.8, control_ok=False),
        _summary(2000, STATE_PATTERN, -0.7, control_ok=False),
    ]
    reversals = detect_state_reversals(rows)
    assert len(reversals) == 1
    assert reversals[0]["status"] == "STATE_REVERSAL_REQUIRES_CONTROL_VALIDATION"
    assert detect_candidate_transitions(rows) == []
