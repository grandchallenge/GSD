from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.control_validation import bootstrap_delta_ci
from gsd.parameter_recovery import adjudicate_r3_control_repair
from gsd.scoring import bootstrap_ci
from run_wp06_r1_activation import (
    EXACT_MODES,
    TARGET,
    TASK_CONFIG,
    build_base_k8_prompts,
    load_model,
    load_tokenizer,
    make_records,
    record_indices,
    score_plain,
    write_json,
)
from run_wp06_r3_parameter import (
    LEARNING_RATE,
    SEEDS,
    TRAIN_ITEMS,
    install_lora,
    train_adapter,
)

CONTROL_N = 128
SENTIMENT_DEMO_K = 16
EXPECTED_DELTA_NORMS = {
    0: 4.36959981918335,
    1: 4.485442161560059,
    2: 4.415067672729492,
}
DELTA_NORM_REL_TOL = 0.05
FINAL_LOSS_MAX = 1e-3


def strip_review(prompt: str) -> str:
    text = prompt
    if text.startswith("Review:"):
        text = text[len("Review:"):]
    if "\nAnswer:" in text:
        text = text.split("\nAnswer:", 1)[0]
    return text.strip()


def build_control_prompts(dataset_repo: str, dataset_sha: str):
    from datasets import load_dataset

    sst2 = list(
        load_dataset(
            dataset_repo,
            "flipped_answer.sst2",
            split="items",
            revision=dataset_sha,
        )
    )
    arithmetic = list(
        load_dataset(
            dataset_repo,
            "successive_answer.arithmetic",
            split="items",
            revision=dataset_sha,
        )
    )

    demos = [x for x in sst2 if x["split"] == "demo"]
    sentiment_tests = [x for x in sst2 if x["split"] == "test"][:CONTROL_N]
    arithmetic_tests = [x for x in arithmetic if x["split"] == "test"][:CONTROL_N]

    rng = random.Random(0)
    d0 = [x for x in demos if int(x["label"]) == 0]
    d1 = [x for x in demos if int(x["label"]) == 1]
    rng.shuffle(d0)
    rng.shuffle(d1)
    chosen = d0[: SENTIMENT_DEMO_K // 2] + d1[: SENTIMENT_DEMO_K // 2]
    rng.shuffle(chosen)
    inverse = {"Positive": "Negative", "Negative": "Positive"}
    demo_blocks = [f"{d['prompt']} {inverse[d['answer']]}" for d in chosen]

    sentiment = (
        ["\n\n".join(demo_blocks + [t["prompt"]]) for t in sentiment_tests],
        [t["correct_answer"] for t in sentiment_tests],
        [t["incorrect_answer"] for t in sentiment_tests],
    )
    arithmetic_control = (
        [t["prompt"] for t in arithmetic_tests],
        [t["correct_answer"] for t in arithmetic_tests],
        [t["incorrect_answer"] for t in arithmetic_tests],
    )
    lm_texts = [strip_review(t["prompt"]) for t in sentiment_tests]
    return sentiment, arithmetic_control, lm_texts


def pair_summary(model, records, pad_id):
    items = list(range(CONTROL_N))
    scores = score_plain(
        model,
        records,
        record_indices(items),
        batch_size=2,
        pad_id=pad_id,
    )
    margins = [scores[2 * i] - scores[2 * i + 1] for i in items]
    lo, hi = bootstrap_ci(margins, n_boot=4000, seed=0)
    return {
        "n": len(margins),
        "margins": margins,
        "mean_margin": sum(margins) / len(margins),
        "ci95": [lo, hi],
    }


def lm_summary(model, tokenizer, texts, pad_id):
    import torch

    means = []
    with torch.inference_mode():
        for off in range(0, len(texts), 4):
            batch_texts = texts[off : off + 4]
            ids_list = [
                tokenizer.encode(x, add_special_tokens=False)
                for x in batch_texts
            ]
            max_len = max(len(x) for x in ids_list)
            input_ids = torch.full(
                (len(ids_list), max_len),
                int(pad_id),
                dtype=torch.long,
                device="cuda",
            )
            attention_mask = torch.zeros(
                (len(ids_list), max_len),
                dtype=torch.long,
                device="cuda",
            )
            for row, ids in enumerate(ids_list):
                input_ids[row, : len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device="cuda"
                )
                attention_mask[row, : len(ids)] = 1

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            for row, ids in enumerate(ids_list):
                if len(ids) < 2:
                    continue
                logits = outputs.logits[row, : len(ids) - 1].float()
                next_tokens = input_ids[row, 1 : len(ids)]
                logp = torch.log_softmax(logits, dim=-1)
                vals = logp.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
                means.append(float(vals.mean().item()))
            del outputs, input_ids, attention_mask

    lo, hi = bootstrap_ci(means, n_boot=4000, seed=0)
    return {
        "n": len(means),
        "mean_token_logprobs": means,
        "mean_token_logprob": sum(means) / len(means),
        "ci95": [lo, hi],
        "perplexity_proxy": math.exp(-sum(means) / len(means)),
    }


def score_controls(model, tokenizer, sentiment_records, arithmetic_records, lm_texts, pad_id):
    return {
        "sentiment": pair_summary(model, sentiment_records, pad_id),
        "arithmetic": pair_summary(model, arithmetic_records, pad_id),
        "language_model": lm_summary(model, tokenizer, lm_texts, pad_id),
    }


def delta_evidence(base: dict, adapted: dict) -> dict:
    out = {}
    degraded_count = 0
    for name in ("sentiment", "arithmetic"):
        delta, lo, hi = bootstrap_delta_ci(
            base[name]["margins"],
            adapted[name]["margins"],
            n_boot=4000,
            seed=0,
        )
        degraded = hi < 0.0
        degraded_count += int(degraded)
        out[name] = {
            "mean_margin_delta": delta,
            "delta_ci95": [lo, hi],
            "resolved_degradation": degraded,
        }

    delta, lo, hi = bootstrap_delta_ci(
        base["language_model"]["mean_token_logprobs"],
        adapted["language_model"]["mean_token_logprobs"],
        n_boot=4000,
        seed=0,
    )
    degraded = hi < 0.0
    degraded_count += int(degraded)
    out["language_model"] = {
        "mean_token_logprob_delta": delta,
        "delta_ci95": [lo, hi],
        "resolved_degradation": degraded,
    }
    return {
        "control_ok": degraded_count == 0,
        "resolved_degradation_count": degraded_count,
        "controls": out,
        "rule": "strict R3C repair: zero of three controls may show resolved degradation",
    }


def reproduction_ok(seed: int, training: dict) -> bool:
    expected = EXPECTED_DELTA_NORMS[seed]
    observed = float(training["delta_frobenius_norm"])
    rel = abs(observed - expected) / expected
    return rel <= DELTA_NORM_REL_TOL and float(training["final_loss"]) < FINAL_LOSS_MAX


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP06-R3C")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP06-R3C")
    hardware = torch.cuda.get_device_name(0)

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    target_sha = resolve_hf_model_sha(args.model, TARGET)

    rows = list(
        load_dataset(
            args.dataset,
            TASK_CONFIG,
            split="items",
            revision=dataset_sha,
        )
    )
    demos = [x for x in rows if x["split"] == "demo"]
    tests = [x for x in rows if x["split"] == "test"][:192]

    tokenizer = load_tokenizer(args.model, TARGET)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must expose a pad token")

    train_prompts, train_correct, train_incorrect = build_base_k8_prompts(
        demos, tests, seed=0
    )
    train_records, train_modes = make_records(
        tokenizer,
        train_prompts,
        train_correct,
        train_incorrect,
    )

    sentiment_triplet, arithmetic_triplet, lm_texts = build_control_prompts(
        args.dataset,
        dataset_sha,
    )
    sentiment_records, sentiment_modes = make_records(tokenizer, *sentiment_triplet)
    arithmetic_records, arithmetic_modes = make_records(tokenizer, *arithmetic_triplet)
    exact_boundary_ok = all(
        mode in EXACT_MODES
        for mode in train_modes + sentiment_modes + arithmetic_modes
    )

    base_model = load_model(args.model, TARGET)
    base_controls = score_controls(
        base_model,
        tokenizer,
        sentiment_records,
        arithmetic_records,
        lm_texts,
        pad_id,
    )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    rows_out = []
    reproduction_flags = []
    control_flags = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        model = load_model(args.model, TARGET)
        lora = install_lora(model, seed=seed)
        training = train_adapter(
            model,
            lora,
            train_records,
            seed=seed,
            reversed_objective=False,
            pad_id=pad_id,
        )
        controls = score_controls(
            model,
            tokenizer,
            sentiment_records,
            arithmetic_records,
            lm_texts,
            pad_id,
        )
        delta = delta_evidence(base_controls, controls)
        repro = reproduction_ok(seed, training)
        reproduction_flags.append(repro)
        control_flags.append(bool(delta["control_ok"]))
        rows_out.append({
            "seed": seed,
            "training": training,
            "reproduction_ok": repro,
            "controls": controls,
            "delta_vs_base": delta,
        })
        del model, lora
        gc.collect()
        torch.cuda.empty_cache()

    adjudication = adjudicate_r3_control_repair(
        exact_boundary_ok=exact_boundary_ok,
        reproduction_ok=reproduction_flags,
        adapter_control_ok=control_flags,
    )

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R3C",
        "protocol": "experiments/gsd-001/WP06_R3_CONTROL_REPAIR_PROTOCOL.md",
        "target_revision": TARGET,
        "hardware": hardware,
        "truthy_reserve_evaluated": False,
        "control_n": CONTROL_N,
        "exact_boundary_ok": exact_boundary_ok,
        "base_controls": base_controls,
        "adapters": rows_out,
        "reproduction_ok": reproduction_flags,
        "adapter_control_ok": control_flags,
        **adjudication,
        "claim_boundary": (
            "control-gate repair only; primary R3 reserve result remains immutable"
        ),
    }

    root = Path(args.output_dir)
    write_json(root / "wp06_r3c_result.json", result)
    out = root / TARGET
    write_json(
        out / "summary.json",
        {
            "revision": TARGET,
            "role": "r3_control_gate_repair",
            "base_controls": base_controls,
            "adapters": rows_out,
            "disposition": result["disposition"],
        },
    )
    write_json(
        out / "manifest.json",
        {
            "programme": "GSD-001",
            "work_package": "GSD-WP06",
            "tier": "R3C",
            "kind": "PARAMETER_LIGHT_CONTROL_GATE_REPAIR",
            "protocol": "experiments/gsd-001/WP06_R3_CONTROL_REPAIR_PROTOCOL.md",
            "model_repository": args.model,
            "model_revision": TARGET,
            "model_resolved_sha": target_sha,
            "dataset_repository": args.dataset,
            "dataset_resolved_sha": dataset_sha,
            "training_rows": "128-191",
            "truthy_reserve_evaluated": False,
            "control_n": CONTROL_N,
            "seeds": list(SEEDS),
            "hardware": hardware,
            "precision": "bfloat16",
            "claim_boundary": (
                "control-gate repair only; no E3 persistence claim"
            ),
        },
    )


if __name__ == "__main__":
    main()
