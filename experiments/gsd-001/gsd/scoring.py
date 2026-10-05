from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

STATE_GENERALIZING = "GENERALIZING"
STATE_PATTERN = "PATTERN_MATCHING"
STATE_UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class SpanScore:
    log_prob: float
    mean_token_log_prob: float
    avg_token_prob: float
    token_count: int


def resolve_answer_start(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], int, str]:
    """Resolve an exact continuation boundary for teacher-forced scoring.

    Prefer the public runner canonical full-text tokenization when the
    tokenization of prompt plus one space is a true prefix. If tokenization
    merges across that boundary, force the boundary explicitly by tokenizing
    the prompt and the leading-space answer continuation separately and
    concatenating their token IDs. The caller must pass the returned token IDs
    directly to the model; re-tokenizing the full text invalidates this contract.
    """
    full_text = prompt + " " + answer
    prefix_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids, len(prefix_ids), "stable_prompt_space"

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    continuation_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not continuation_ids:
        raise ValueError("answer continuation tokenized to an empty sequence")

    forced_ids = list(prompt_ids) + list(continuation_ids)
    return forced_ids, len(prompt_ids), "forced_separate_continuation"


def span_score_from_token_logprobs(logps: Iterable[float]) -> SpanScore:
    values = [float(x) for x in logps]
    if not values:
        return SpanScore(float("-inf"), float("-inf"), 0.0, 0)
    total = float(sum(values))
    mean = total / len(values)
    avg_prob = float(sum(math.exp(x) for x in values) / len(values))
    return SpanScore(total, mean, avg_prob, len(values))


def record_from_pair_scores(
    intelligence: SpanScore,
    parrot: SpanScore,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        **metadata,
        "intelligence": asdict(intelligence),
        "parrot": asdict(parrot),
        "sum_margin": intelligence.log_prob - parrot.log_prob,
        "mean_token_margin": intelligence.mean_token_log_prob - parrot.mean_token_log_prob,
        "avg_token_prob_margin": intelligence.avg_token_prob - parrot.avg_token_prob,
        "upstream_hard_generalizes": intelligence.avg_token_prob > parrot.avg_token_prob,
    }


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        x = float(arr[0])
        return x, x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def classify_margins(
    values: Iterable[float],
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    vals = [float(x) for x in values]
    if not vals:
        return {
            "state": STATE_UNCERTAIN,
            "n": 0,
            "mean_margin": float("nan"),
            "median_margin": float("nan"),
            "positive_fraction": float("nan"),
            "ci95": [float("nan"), float("nan")],
        }
    lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
    if lo > 0:
        state = STATE_GENERALIZING
    elif hi < 0:
        state = STATE_PATTERN
    else:
        state = STATE_UNCERTAIN
    arr = np.asarray(vals, dtype=float)
    return {
        "state": state,
        "n": len(vals),
        "mean_margin": float(arr.mean()),
        "median_margin": float(np.median(arr)),
        "positive_fraction": float((arr > 0).mean()),
        "ci95": [lo, hi],
    }


def score_pairs_hf(
    model: Any,
    tokenizer: Any,
    pairs: list[tuple[str, str]],
    *,
    batch_size: int = 8,
) -> list[tuple[SpanScore, str]]:
    """Teacher-force answer spans with a causal Hugging Face model."""
    import torch

    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    encoded = [resolve_answer_start(tokenizer, p, a) for p, a in pairs]
    results: list[tuple[SpanScore, str]] = []

    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        max_len = max(len(ids) for ids, _, _ in chunk)
        pad = tokenizer.pad_token_id
        input_ids = []
        attention = []
        for ids, _, _ in chunk:
            n_pad = max_len - len(ids)
            input_ids.append(ids + [pad] * n_pad)
            attention.append([1] * len(ids) + [0] * n_pad)
        ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(attention, dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids_t, attention_mask=mask_t).logits
            logp = torch.log_softmax(logits.float(), dim=-1)

        for row, (ids, ans_start, boundary_mode) in enumerate(chunk):
            token_logps: list[float] = []
            for pos in range(ans_start, len(ids)):
                if pos == 0:
                    continue
                token_logps.append(float(logp[row, pos - 1, ids[pos]].item()))
            results.append((span_score_from_token_logprobs(token_logps), boundary_mode))

    return results
