from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .scoring import STATE_GENERALIZING, STATE_PATTERN, classify_margins


@dataclass(frozen=True)
class CheckpointSummary:
    revision: str
    step: int
    state: str
    mean_margin: float
    ci_low: float
    ci_high: float
    n: int
    control_ok: bool = False


def summarize_checkpoint(
    revision: str,
    step: int,
    margins: Iterable[float],
    *,
    control_ok: bool = False,
    n_boot: int = 4000,
    seed: int = 0,
) -> CheckpointSummary:
    summary = classify_margins(margins, n_boot=n_boot, seed=seed)
    return CheckpointSummary(
        revision=revision,
        step=step,
        state=summary["state"],
        mean_margin=summary["mean_margin"],
        ci_low=summary["ci95"][0],
        ci_high=summary["ci95"][1],
        n=summary["n"],
        control_ok=control_ok,
    )


def detect_state_reversals(
    summaries: Iterable[CheckpointSummary],
) -> list[dict[str, Any]]:
    """Find adjacent confident sign/state reversals before control validation."""
    ordered = sorted(summaries, key=lambda s: s.step)
    out: list[dict[str, Any]] = []
    confident = {STATE_GENERALIZING, STATE_PATTERN}
    for left, right in zip(ordered, ordered[1:]):
        if left.state not in confident or right.state not in confident:
            continue
        if left.state == right.state:
            continue
        out.append(
            {
                "from_revision": left.revision,
                "to_revision": right.revision,
                "from_step": left.step,
                "to_step": right.step,
                "from_state": left.state,
                "to_state": right.state,
                "step_gap": right.step - left.step,
                "control_validated": bool(left.control_ok and right.control_ok),
                "status": "STATE_REVERSAL_REQUIRES_CONTROL_VALIDATION",
            }
        )
    return out


def detect_candidate_transitions(
    summaries: Iterable[CheckpointSummary],
) -> list[dict[str, Any]]:
    """Promote only reversals whose two endpoints have stable-control evidence."""
    out: list[dict[str, Any]] = []
    for reversal in detect_state_reversals(summaries):
        if not reversal["control_validated"]:
            continue
        promoted = dict(reversal)
        promoted["status"] = "CANDIDATE_TRANSITION"
        out.append(promoted)
    return out


def variant_agreement(base_direction: int, variant_margins: Iterable[float]) -> float:
    vals = list(variant_margins)
    if not vals:
        return 0.0
    return sum(1 for x in vals if (x > 0) == (base_direction > 0)) / len(vals)


def adversarial_disposition(
    *,
    base_margin: float,
    prompt_variant_margins: Iterable[float],
    token_boundary_stable: bool,
    soft_margin_resolved: bool,
    control_ok: bool,
    min_prompt_agreement: float = 0.8,
) -> str:
    if not token_boundary_stable:
        return "TOKEN_BOUNDARY_ARTEFACT_RISK"
    if not soft_margin_resolved:
        return "THRESHOLD_OR_UNCERTAINTY_ARTEFACT"
    if not control_ok:
        return "GENERIC_DEGRADATION_NOT_GSD"
    agreement = variant_agreement(1 if base_margin > 0 else -1, prompt_variant_margins)
    if agreement < min_prompt_agreement:
        return "PROMPT_SENSITIVE_UNCONFIRMED"
    return "SURVIVES_ADVERSARIAL_CHECKS"
