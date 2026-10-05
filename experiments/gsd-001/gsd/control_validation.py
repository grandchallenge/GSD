from __future__ import annotations

import math
import random
from statistics import mean


def bootstrap_delta_ci(
    left: list[float],
    right: list[float],
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired control vectors must be non-empty and equal length")
    deltas = [r - l for l, r in zip(left, right)]
    rng = random.Random(seed)
    boots = []
    n = len(deltas)
    for _ in range(n_boot):
        boots.append(mean(deltas[rng.randrange(n)] for _ in range(n)))
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    return mean(deltas), lo, hi


def paired_control_disposition(
    left: dict,
    right: dict,
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict:
    evidence = {}
    resolved_degradations = 0
    catastrophic = False

    for name in ("sentiment", "arithmetic"):
        a = left[name]
        b = right[name]
        delta, lo, hi = bootstrap_delta_ci(
            a["margins"], b["margins"], n_boot=n_boot, seed=seed
        )
        hard_delta = float(b["hard_accuracy"]) - float(a["hard_accuracy"])
        degraded = hi < 0.0 and hard_delta < 0.0
        catastrophic_here = hard_delta <= -0.15
        resolved_degradations += int(degraded)
        catastrophic = catastrophic or catastrophic_here
        evidence[name] = {
            "mean_margin_delta": delta,
            "delta_ci95": [lo, hi],
            "hard_accuracy_delta": hard_delta,
            "resolved_degradation": degraded,
            "catastrophic": catastrophic_here,
        }

    a = left["language_model"]
    b = right["language_model"]
    delta, lo, hi = bootstrap_delta_ci(
        a["mean_token_logprobs"], b["mean_token_logprobs"],
        n_boot=n_boot, seed=seed
    )
    degraded = hi < 0.0
    catastrophic_here = delta <= -0.15
    resolved_degradations += int(degraded)
    catastrophic = catastrophic or catastrophic_here
    evidence["language_model"] = {
        "mean_token_logprob_delta": delta,
        "delta_ci95": [lo, hi],
        "resolved_degradation": degraded,
        "catastrophic": catastrophic_here,
    }

    control_ok = (not catastrophic) and resolved_degradations <= 1
    return {
        "control_ok": control_ok,
        "resolved_degradation_count": resolved_degradations,
        "catastrophic_degradation": catastrophic,
        "controls": evidence,
        "rule": (
            "fail if any catastrophic ordinary-capability degradation occurs or "
            "if at least two of three independent controls show resolved degradation"
        ),
    }
