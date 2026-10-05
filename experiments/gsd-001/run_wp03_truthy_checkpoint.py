from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.scoring import classify_margins, resolve_answer_start


TASKS = {
    "truthy_answer/surprising_truth": {
        "config": "truthy_answer.surprising_truth",
        "k": 8,
    },
    "truthy_answer/common_misconception": {
        "config": "truthy_answer.common_misconception",
        "k": 100,
    },
}


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


def transform_markers(text: str) -> str:
    text = re.sub(r"(?m)^Q:", "Question:", text)
    text = re.sub(r"(?m)^A:", "Answer:", text)
    return text


def build_prompts(
    demos: list[dict],
    tests: list[dict],
    *,
    k: int,
    seed: int,
    variant: str,
) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    prompts: list[str] = []
    correct: list[str] = []
    incorrect: list[str] = []

    for test in tests:
        chosen = rng.sample(demos, min(k, len(demos)))
        blocks = [f"{d['prompt']} {d['answer']}" for d in chosen]
        test_block = test["prompt"]
        joiner = "\n"

        if variant == "expanded_markers":
            blocks = [transform_markers(x) for x in blocks]
            test_block = transform_markers(test_block)
        elif variant == "double_newline":
            joiner = "\n\n"
        elif variant == "instruction_prefix":
            pass
        elif variant != "base":
            raise ValueError(f"unknown prompt variant: {variant}")

        prompt = joiner.join(blocks + [test_block])
        if variant == "instruction_prefix":
            prompt = (
                "Determine whether each statement is True or False. "
                "Answer only True or False.\n" + prompt
            )

        prompts.append(prompt)
        correct.append(test["correct_answer"])
        incorrect.append(test["incorrect_answer"])

    return prompts, correct, incorrect


def score_answers(llm, tokenizer, prompts: list[str], answers: list[str]) -> tuple[list[dict], list[str]]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    token_prompts = []
    expected_ids: list[list[int]] = []
    starts: list[int] = []
    modes: list[str] = []
    for prompt, answer in zip(prompts, answers):
        full_ids, start, mode = resolve_answer_start(tokenizer, prompt, answer)
        ids = list(full_ids)
        token_prompts.append(TokensPrompt(prompt_token_ids=ids))
        expected_ids.append(ids)
        starts.append(start)
        modes.append(mode)

    outputs = llm.generate(
        token_prompts,
        SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0),
        use_tqdm=True,
    )

    result: list[dict] = []
    for out, start, expected in zip(outputs, starts, expected_ids):
        seq = out.prompt_logprobs
        ids = list(out.prompt_token_ids)
        if ids != expected:
            raise RuntimeError("vLLM token prompt drifted from exact boundary IDs")
        logs: list[float] = []
        probs: list[float] = []
        if seq is not None:
            for pos in range(start, len(seq)):
                cell = seq[pos]
                if cell is None or pos >= len(ids):
                    continue
                tid = ids[pos]
                if tid in cell:
                    lp = float(cell[tid].logprob)
                    logs.append(lp)
                    probs.append(math.exp(lp))
        result.append({
            "log_prob": sum(logs) if logs else float("-inf"),
            "avg_prob": sum(probs) / len(probs) if probs else 0.0,
            "token_count": len(logs),
        })
    return result, modes


def score_pair_set(llm, tokenizer, prompts, correct, incorrect) -> dict:
    c, c_modes = score_answers(llm, tokenizer, prompts, correct)
    i, i_modes = score_answers(llm, tokenizer, prompts, incorrect)
    margins = [a["log_prob"] - b["log_prob"] for a, b in zip(c, i)]
    summary = classify_margins(margins, n_boot=4000, seed=0)
    hard = sum(
        1 for a, b in zip(c, i) if a["avg_prob"] > b["avg_prob"]
    ) / max(1, len(c))
    all_modes = c_modes + i_modes
    return {
        **summary,
        "hard_generalizes_rate": hard,
        "margins": margins,
        "boundary_mode_counts": {
            mode: all_modes.count(mode) for mode in sorted(set(all_modes))
        },
        "boundary_all_stable": all(
            mode == "stable_prompt_space" for mode in all_modes
        ),
        "boundary_all_exact": all(
            mode in {"stable_prompt_space", "forced_separate_continuation"}
            for mode in all_modes
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--revision", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["instruction_prefix", "expanded_markers", "double_newline"],
    )
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP03")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP03")

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    model_sha = resolve_hf_model_sha(args.model, args.revision)
    llm = build_llm(args.model, args.revision)
    tokenizer = llm.get_tokenizer()

    task_outputs = {}
    all_boundaries_stable = True
    all_boundaries_exact = True

    for task_key, task_cfg in TASKS.items():
        rows = list(load_dataset(
            args.dataset,
            task_cfg["config"],
            split="items",
            revision=dataset_sha,
        ))
        demos = [x for x in rows if x["split"] == "demo"]
        tests = [x for x in rows if x["split"] == "test"][:args.max_eval]

        seed_outputs = {}
        pooled_margins: list[float] = []
        task_boundary_stable = True
        task_boundary_exact = True

        for seed in args.seeds:
            prompts, correct, incorrect = build_prompts(
                demos, tests, k=task_cfg["k"], seed=seed, variant="base"
            )
            result = score_pair_set(llm, tokenizer, prompts, correct, incorrect)
            pooled_margins.extend(result["margins"])
            task_boundary_stable = task_boundary_stable and result["boundary_all_stable"]
            task_boundary_exact = task_boundary_exact and result["boundary_all_exact"]
            seed_outputs[str(seed)] = {
                k: v for k, v in result.items() if k != "margins"
            }

        pooled = classify_margins(pooled_margins, n_boot=4000, seed=0)
        variants = {}
        for variant in args.variants:
            prompts, correct, incorrect = build_prompts(
                demos, tests, k=task_cfg["k"], seed=0, variant=variant
            )
            result = score_pair_set(llm, tokenizer, prompts, correct, incorrect)
            task_boundary_stable = task_boundary_stable and result["boundary_all_stable"]
            task_boundary_exact = task_boundary_exact and result["boundary_all_exact"]
            variants[variant] = {
                k: v for k, v in result.items() if k != "margins"
            }

        all_boundaries_stable = all_boundaries_stable and task_boundary_stable
        all_boundaries_exact = all_boundaries_exact and task_boundary_exact
        task_outputs[task_key] = {
            "base_replay": {
                "seeds": seed_outputs,
                "pooled": pooled,
            },
            "prompt_variants_seed0": variants,
            "boundary_all_stable": task_boundary_stable,
            "boundary_all_exact": task_boundary_exact,
        }

    step = int(args.revision.split("step", 1)[1].split("-", 1)[0])
    out = Path(args.output_dir) / args.revision
    summary = {
        "revision": args.revision,
        "step": step,
        "tasks": task_outputs,
        "boundary_all_stable": all_boundaries_stable,
        "boundary_all_exact": all_boundaries_exact,
    }
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {
        "programme": "GSD-001",
        "work_package": "GSD-WP03",
        "kind": "TRUTHY_ADVERSARIAL_REPLAY_BOUNDARY_V2",
        "boundary_contract": "canonical_prefix_else_forced_separate_continuation_v1",
        "model_repository": args.model,
        "model_revision": args.revision,
        "model_resolved_sha": model_sha,
        "dataset_repository": args.dataset,
        "dataset_resolved_sha": dataset_sha,
        "seeds": args.seeds,
        "variants": args.variants,
        "max_eval": args.max_eval,
        "hardware": torch.cuda.get_device_name(0),
        "precision": "bfloat16",
        "claim_boundary": "behavioural transition validation only; no mechanism claim",
    })

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
