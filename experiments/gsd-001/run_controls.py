from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.control_validation import paired_control_disposition
from gsd.scoring import bootstrap_ci


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_llm(model: str, revision: str):
    from vllm import LLM
    return LLM(
        model=model,
        revision=revision,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
    )


def cleanup_llm(llm) -> None:
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def score_answers(llm, tokenizer, prompts: list[str], answers: list[str]) -> list[dict]:
    from vllm import SamplingParams
    full_texts = []
    answer_starts = []
    for prompt, answer in zip(prompts, answers):
        full = prompt + " " + answer
        prompt_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        start = len(prompt_ids)
        if full_ids[:start] != prompt_ids:
            start = len(tokenizer.encode(prompt, add_special_tokens=False))
        full_texts.append(full)
        answer_starts.append(start)

    outputs = llm.generate(
        full_texts,
        SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0),
        use_tqdm=True,
    )
    result = []
    for out, start in zip(outputs, answer_starts):
        logs = []
        probs = []
        seq = out.prompt_logprobs
        ids = out.prompt_token_ids
        if seq is not None:
            for pos in range(start, len(seq)):
                cell = seq[pos]
                if cell is None or pos >= len(ids):
                    continue
                tid = ids[pos]
                if tid in cell:
                    lp = cell[tid].logprob
                    logs.append(lp)
                    probs.append(math.exp(lp))
        result.append({
            "log_prob": sum(logs) if logs else float("-inf"),
            "avg_prob": sum(probs) / len(probs) if probs else 0.0,
        })
    return result


def score_pairs(llm, tokenizer, prompts, correct, incorrect) -> dict:
    c = score_answers(llm, tokenizer, prompts, correct)
    i = score_answers(llm, tokenizer, prompts, incorrect)
    margins = [a["log_prob"] - b["log_prob"] for a, b in zip(c, i)]
    hard = [1.0 if a["avg_prob"] > b["avg_prob"] else 0.0 for a, b in zip(c, i)]
    lo, hi = bootstrap_ci(margins, n_boot=4000, seed=0)
    return {
        "n": len(margins),
        "margins": margins,
        "mean_margin": sum(margins) / len(margins),
        "ci95": [lo, hi],
        "hard_accuracy": sum(hard) / len(hard),
    }


def score_texts(llm, texts: list[str]) -> dict:
    from vllm import SamplingParams
    outputs = llm.generate(
        texts,
        SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0),
        use_tqdm=True,
    )
    means = []
    for out in outputs:
        seq = out.prompt_logprobs
        ids = out.prompt_token_ids
        logs = []
        if seq is not None:
            for pos in range(1, len(seq)):
                cell = seq[pos]
                if cell is None or pos >= len(ids):
                    continue
                tid = ids[pos]
                if tid in cell:
                    logs.append(cell[tid].logprob)
        if logs:
            means.append(sum(logs) / len(logs))
    lo, hi = bootstrap_ci(means, n_boot=4000, seed=0)
    return {
        "n": len(means),
        "mean_token_logprobs": means,
        "mean_token_logprob": sum(means) / len(means),
        "ci95": [lo, hi],
        "perplexity_proxy": math.exp(-sum(means) / len(means)),
    }


def strip_review(prompt: str) -> str:
    text = prompt
    if text.startswith("Review:"):
        text = text[len("Review:"):]
    if "\nAnswer:" in text:
        text = text.split("\nAnswer:", 1)[0]
    return text.strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--revision", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument("--sentiment-demo-k", type=int, default=16)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for stable-control validation")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (8, 0):
        raise SystemExit("BF16-capable GPU (compute capability >= 8.0) required")

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    sst2 = list(load_dataset(
        args.dataset, "flipped_answer.sst2",
        split="items", revision=dataset_sha
    ))
    arithmetic = list(load_dataset(
        args.dataset, "successive_answer.arithmetic",
        split="items", revision=dataset_sha
    ))

    demos = [x for x in sst2 if x["split"] == "demo"]
    sentiment_tests = [x for x in sst2 if x["split"] == "test"][:args.max_eval]
    arithmetic_tests = [x for x in arithmetic if x["split"] == "test"][:args.max_eval]

    rng = random.Random(0)
    d0 = [x for x in demos if int(x["label"]) == 0]
    d1 = [x for x in demos if int(x["label"]) == 1]
    rng.shuffle(d0); rng.shuffle(d1)
    chosen = d0[:args.sentiment_demo_k // 2] + d1[:args.sentiment_demo_k // 2]
    rng.shuffle(chosen)
    inverse = {"Positive": "Negative", "Negative": "Positive"}
    demo_blocks = [f"{d['prompt']} {inverse[d['answer']]}" for d in chosen]

    sentiment_prompts = [
        "\n\n".join(demo_blocks + [t["prompt"]]) for t in sentiment_tests
    ]
    sentiment_correct = [t["correct_answer"] for t in sentiment_tests]
    sentiment_incorrect = [t["incorrect_answer"] for t in sentiment_tests]

    arithmetic_prompts = [t["prompt"] for t in arithmetic_tests]
    arithmetic_correct = [t["correct_answer"] for t in arithmetic_tests]
    arithmetic_incorrect = [t["incorrect_answer"] for t in arithmetic_tests]

    lm_texts = [strip_review(t["prompt"]) for t in sentiment_tests]

    root = Path(args.output_dir)
    summaries = {}
    for revision in args.revision:
        out = root / revision
        model_sha = resolve_hf_model_sha(args.model, revision)
        llm = build_llm(args.model, revision)
        tok = llm.get_tokenizer()

        sentiment = score_pairs(
            llm, tok, sentiment_prompts, sentiment_correct, sentiment_incorrect
        )
        arithmetic_control = score_pairs(
            llm, tok, arithmetic_prompts, arithmetic_correct, arithmetic_incorrect
        )
        language_model = score_texts(llm, lm_texts)

        step = int(revision.split("step", 1)[1].split("-", 1)[0])
        summary = {
            "revision": revision,
            "step": step,
            "controls": {
                "sentiment": sentiment,
                "arithmetic": arithmetic_control,
                "language_model": language_model,
            },
        }
        summaries[revision] = summary
        write_json(out / "summary.json", summary)
        write_json(out / "manifest.json", {
            "programme": "GSD-001",
            "work_package": "GSD-WP02",
            "kind": "STABLE_CONTROL_VALIDATION",
            "model_repository": args.model,
            "model_revision": revision,
            "model_resolved_sha": model_sha,
            "dataset_repository": args.dataset,
            "dataset_resolved_sha": dataset_sha,
            "sentiment_config": "flipped_answer.sst2 reconstructed with correct-label demonstrations",
            "sentiment_demo_k": args.sentiment_demo_k,
            "arithmetic_config": "successive_answer.arithmetic evaluated zero-shot without successive demonstrations",
            "language_model_config": "held-out SST-2 review text mean token log-likelihood",
            "max_eval": args.max_eval,
            "hardware": torch.cuda.get_device_name(0),
            "precision": "bfloat16",
            "claim_boundary": "stable-control evidence only; no mechanism claim",
        })
        cleanup_llm(llm)

    ordered = sorted(summaries.values(), key=lambda x: x["step"])
    pair_map = {}
    pair_evidence = {}
    for left, right in zip(ordered, ordered[1:]):
        key = f"{left['revision']}->{right['revision']}"
        disposition = paired_control_disposition(
            left["controls"], right["controls"], n_boot=4000, seed=0
        )
        pair_map[key] = bool(disposition["control_ok"])
        pair_evidence[key] = disposition

    write_json(root / "pair_controls.json", pair_map)
    write_json(root / "pair_control_evidence.json", pair_evidence)


if __name__ == "__main__":
    main()
