from __future__ import annotations

from dataclasses import dataclass


CONFIRMED = "CONFIRMED_TRANSITION"
TOKEN_RISK = "TOKEN_BOUNDARY_ARTEFACT_RISK"
PROMPT_RISK = "PROMPT_SENSITIVE_UNCONFIRMED"
REPLAY_FAIL = "INDEPENDENT_REPLAY_FAILED"
CONTROL_FAIL = "GENERIC_DEGRADATION_NOT_GSD"


@dataclass(frozen=True)
class CandidateSpec:
    task: str
    from_revision: str
    to_revision: str
    from_state: str
    to_state: str


def validate_candidate(
    spec: CandidateSpec,
    *,
    control_ok: bool,
    endpoint_boundaries_stable: bool,
    variant_endpoint_states: dict[str, tuple[str, str]],
    replay_endpoint_states: dict[int, tuple[str, str]],
    min_replay_seeds: int = 3,
) -> dict:
    if not control_ok:
        return {
            "disposition": CONTROL_FAIL,
            "replay_support": 0,
            "required_replay_support": min_replay_seeds,
        }
    if not endpoint_boundaries_stable:
        return {
            "disposition": TOKEN_RISK,
            "replay_support": 0,
            "required_replay_support": min_replay_seeds,
        }

    expected = (spec.from_state, spec.to_state)
    prompt_failures = {
        name: states
        for name, states in variant_endpoint_states.items()
        if states != expected
    }
    if prompt_failures:
        return {
            "disposition": PROMPT_RISK,
            "prompt_failures": prompt_failures,
            "replay_support": 0,
            "required_replay_support": min_replay_seeds,
        }

    replay_support = sum(
        1 for states in replay_endpoint_states.values() if states == expected
    )
    if replay_support < min_replay_seeds:
        return {
            "disposition": REPLAY_FAIL,
            "replay_support": replay_support,
            "required_replay_support": min_replay_seeds,
        }

    return {
        "disposition": CONFIRMED,
        "replay_support": replay_support,
        "required_replay_support": min_replay_seeds,
        "prompt_variant_count": len(variant_endpoint_states),
    }
