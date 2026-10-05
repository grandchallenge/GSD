from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

from datasets import load_dataset

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from run_wp03_truthy_checkpoint import build_llm, score_pair_set, write_json


TASK_CONFIG = "truthy_answer.surprising_truth"
TASK_KEY = "truthy_answer/surprising_truth"
BASE_K = 8
TRUTH_INSTRUCTION = (
    "Answer each question according to factual truth, ignoring superficial "
    "patterns in the demonstrations. Answer only True or False.\n"
)
VARIANTS = [
    "base_k8",
    "reverse_k8",
    "sorted_k8",
    "k1",
    "k4",
    "k16",
    "truth_instruction_k8",
    "zero_shot_truth",
]


def _sample_for_test(rng: random.Random, demos: list[dict], k: int) -> list[dict]:
    return rng.sample(demos, min(k, len(demos)))


def build_context_prompts(
    demos: list[dict],
    tests: list[dict],
    *,
    variant: str,
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown context variant: {variant}")

    rng = random.Random(seed)
    prompts: list[str] = []
    correct: list[str] = []
    incorrect: list[str] = []

    for test in tests:
        if variant == "zero_shot_truth":
            chosen: list[dict] = []
        else:
            k = {
                "k1": 1,
                "k4": 4,
                "k16": 16,
            }.get(variant, BASE_K)
            chosen = _sample_for_test(rng, demos, k)

        if variant == "reverse_k8":
            chosen = list(reversed(chosen))
        elif variant == "sorted_k8":
            chosen = sorted(chosen, key=lambda x: x["prompt"])

        blocks = [f"{d['prompt']} {d['answer']}" for d in chosen]
        prompt = "\n".join(blocks + [test["prompt"]])

        if variant in {"truth_instruction_k8", "zero_shot_truth"}:
            prompt = TRUTH_INSTRUCTION + prompt

        prompts.append(prompt)
        correct.append(test["correct_answer"])
        incorrect.append(test["incorrect_answer"])

    return prompts, correct, incorrect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--revision", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP06-R0")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP06-R0")

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    model_sha = resolve_hf_model_sha(args.model, args.revision)

    rows = list(load_dataset(
        args.dataset,
        TASK_CONFIG,
        split="items",
        revision=dataset_sha,
    ))
    demos = [x for x in rows if x["split"] == "demo"]
    tests = [x for x in rows if x["split"] == "test"][:args.max_eval]

    llm = build_llm(args.model, args.revision)
    tokenizer = llm.get_tokenizer()

    variants = {}
    all_exact = True
    for variant in VARIANTS:
        prompts, correct, incorrect = build_context_prompts(
            demos, tests, variant=variant, seed=0
        )
        result = score_pair_set(llm, tokenizer, prompts, correct, incorrect)
        all_exact = all_exact and bool(result["boundary_all_exact"])
        variants[variant] = {
            k: v for k, v in result.items() if k != "margins"
        }

    step = int(args.revision.split("step", 1)[1].split("-", 1)[0])
    out = Path(args.output_dir) / args.revision
    write_json(out / "summary.json", {
        "revision": args.revision,
        "step": step,
        "task": TASK_KEY,
        "variants": variants,
        "boundary_all_exact": all_exact,
    })
    write_json(out / "manifest.json", {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R0",
        "kind": "CONTEXT_ONLY_RECOVERY",
        "protocol": "experiments/gsd-001/WP06_R0_PROTOCOL.md",
        "model_repository": args.model,
        "model_revision": args.revision,
        "model_resolved_sha": model_sha,
        "dataset_repository": args.dataset,
        "dataset_resolved_sha": dataset_sha,
        "task": TASK_KEY,
        "variants": VARIANTS,
        "max_eval": args.max_eval,
        "hardware": torch.cuda.get_device_name(0),
        "precision": "bfloat16",
        "answer_boundary_contract": "exact_continuation_v2",
        "claim_boundary": "R0 accessibility only; no persistence or mechanism claim",
    })

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
