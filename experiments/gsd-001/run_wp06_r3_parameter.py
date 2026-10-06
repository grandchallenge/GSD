from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.parameter_recovery import adjudicate_r3
from run_wp06_r1_activation import (
    EXACT_MODES,
    SOURCE,
    TARGET,
    TASK_CONFIG,
    TASK_KEY,
    build_base_k8_prompts,
    collate,
    load_model,
    load_tokenizer,
    make_records,
    record_indices,
    score_plain,
    summarize,
    verify_tokenizer,
    write_json,
)

TRAIN_ITEMS = list(range(128, 192))
RESERVE_ITEMS = list(range(256, 287))
SEEDS = (0, 1, 2)
TRAIN_STEPS = 32
BATCH_ITEMS = 2
LEARNING_RATE = 3e-3
RANK = 1
ALPHA = 1.0
CONTROL_N = 32
SENTIMENT_DEMO_K = 16


class RankOneLoRA:
    def __new__(cls, base, *, seed: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _LoRA(nn.Module):
            def __init__(self, base_layer, seed_value):
                super().__init__()
                self.base = base_layer
                self.rank = RANK
                self.alpha = ALPHA
                g = torch.Generator(device="cpu")
                g.manual_seed(seed_value)
                a = torch.randn(
                    (RANK, base_layer.in_features),
                    generator=g,
                    dtype=torch.float32,
                ) * 0.01
                b = torch.zeros(
                    (base_layer.out_features, RANK),
                    dtype=torch.float32,
                )
                self.A = nn.Parameter(a.to(base_layer.weight.device))
                self.B = nn.Parameter(b.to(base_layer.weight.device))

            def forward(self, x):
                base_out = self.base(x)
                z = F.linear(x.float(), self.A)
                delta = F.linear(z, self.B) * (self.alpha / self.rank)
                return base_out + delta.to(dtype=base_out.dtype)

            def delta_weight(self):
                return (self.B @ self.A) * (self.alpha / self.rank)

        return _LoRA(base, seed)


def install_lora(model, *, seed: int):
    for p in model.parameters():
        p.requires_grad_(False)
    block = model.model.layers[11]
    base = block.self_attn.o_proj
    lora = RankOneLoRA(base, seed=seed)
    block.self_attn.o_proj = lora
    for p in lora.parameters():
        p.requires_grad_(False)
    lora.A.requires_grad_(True)
    lora.B.requires_grad_(True)
    return lora


def differentiable_sequence_scores(model, records, indices, pad_id):
    import torch

    input_ids, attention_mask = collate(records, indices, pad_id)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    seq_scores = []
    for row, idx in enumerate(indices):
        positions = records[idx]["pred_positions"]
        pos = torch.tensor(
            positions,
            dtype=torch.long,
            device=outputs.logits.device,
        )
        selected = outputs.logits[row].index_select(0, pos).float()
        next_tokens = input_ids[row].index_select(0, pos + 1)
        logp = torch.log_softmax(selected, dim=-1)
        token_logp = logp.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
        seq_scores.append(token_logp.sum())
    return torch.stack(seq_scores)


def train_adapter(
    model,
    lora,
    records,
    *,
    seed: int,
    reversed_objective: bool,
    pad_id: int,
):
    import torch
    import torch.nn.functional as F

    order = list(TRAIN_ITEMS)
    random.Random(seed).shuffle(order)
    optimizer = torch.optim.AdamW(
        [lora.A, lora.B],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    losses = []
    model.eval()
    for step in range(TRAIN_STEPS):
        items = order[step * BATCH_ITEMS : (step + 1) * BATCH_ITEMS]
        if len(items) != BATCH_ITEMS:
            raise RuntimeError("frozen R3 one-epoch batch schedule drifted")
        indices = []
        for item in items:
            indices.extend([2 * item, 2 * item + 1])

        optimizer.zero_grad(set_to_none=True)
        seq_scores = differentiable_sequence_scores(model, records, indices, pad_id)
        correct = seq_scores[0::2]
        incorrect = seq_scores[1::2]
        margin = correct - incorrect
        if reversed_objective:
            margin = -margin
        loss = F.softplus(-margin).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([lora.A, lora.B], max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    delta = lora.delta_weight().detach().float().cpu()
    return {
        "steps": TRAIN_STEPS,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "delta_frobenius_norm": float(delta.norm().item()),
        "a_norm": float(lora.A.detach().float().norm().cpu().item()),
        "b_norm": float(lora.B.detach().float().norm().cpu().item()),
    }


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
    demo_blocks = [
        f"{d['prompt']} {inverse[d['answer']]}"
        for d in chosen
    ]

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
    return sentiment, arithmetic_control


def score_control_family(model, records):
    items = list(range(CONTROL_N))
    indices = record_indices(items)
    scores = score_plain(
        model,
        records,
        indices,
        batch_size=2,
        pad_id=model.config.pad_token_id,
    )
    return summarize(scores, items)


def adapter_manifest(model, lora):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "rank": RANK,
        "alpha": ALPHA,
        "module": "model.layers.11.self_attn.o_proj",
        "layer_one_indexed": 12,
        "trainable_parameters": trainable,
        "total_parameters_with_adapter": total,
        "trainable_fraction": trainable / total,
        "expected_trainable_parameters": 4096,
        "delta_shape": list(lora.delta_weight().shape),
    }


def save_adapter(path: Path, lora, metadata: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "A": lora.A.detach().float().cpu(),
            "B": lora.B.detach().float().cpu(),
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP06-R3")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP06-R3")
    hardware = torch.cuda.get_device_name(0)

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    source_sha = resolve_hf_model_sha(args.model, SOURCE)
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
    tests = [x for x in rows if x["split"] == "test"]
    if len(tests) != 287:
        raise RuntimeError(f"R3 frozen contract requires 287 test rows, got {len(tests)}")

    source_tokenizer = load_tokenizer(args.model, SOURCE)
    pad_id = source_tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must expose a pad token")
    prompts, correct, incorrect = build_base_k8_prompts(demos, tests, seed=0)
    records, truth_modes = make_records(
        source_tokenizer,
        prompts,
        correct,
        incorrect,
    )
    truth_exact = all(mode in EXACT_MODES for mode in truth_modes)
    reserve_indices = record_indices(RESERVE_ITEMS)

    sentiment_triplet, arithmetic_triplet = build_control_prompts(
        args.dataset,
        dataset_sha,
    )
    sentiment_records, sentiment_modes = make_records(
        source_tokenizer,
        *sentiment_triplet,
    )
    arithmetic_records, arithmetic_modes = make_records(
        source_tokenizer,
        *arithmetic_triplet,
    )
    controls_exact = all(
        mode in EXACT_MODES
        for mode in sentiment_modes + arithmetic_modes
    )
    exact_boundary_ok = truth_exact and controls_exact

    output = Path(args.output_dir)

    source_model = load_model(args.model, SOURCE)
    source_reserve_scores = score_plain(
        source_model,
        records,
        reserve_indices,
        batch_size=2,
        pad_id=pad_id,
    )
    source_reserve = summarize(source_reserve_scores, RESERVE_ITEMS)
    del source_model
    gc.collect()
    torch.cuda.empty_cache()

    target_tokenizer = load_tokenizer(args.model, TARGET)
    verify_tokenizer(records, target_tokenizer, prompts, correct, incorrect)
    verify_tokenizer(
        sentiment_records,
        target_tokenizer,
        *sentiment_triplet,
    )
    verify_tokenizer(
        arithmetic_records,
        target_tokenizer,
        *arithmetic_triplet,
    )

    base_model = load_model(args.model, TARGET)
    base_reserve_scores = score_plain(
        base_model,
        records,
        reserve_indices,
        batch_size=2,
        pad_id=pad_id,
    )
    target_reserve = summarize(base_reserve_scores, RESERVE_ITEMS)
    base_sentiment = score_control_family(base_model, sentiment_records)
    base_arithmetic = score_control_family(base_model, arithmetic_records)
    base_controls_ok = (
        base_sentiment["state"] == "GENERALIZING"
        and base_arithmetic["state"] == "GENERALIZING"
    )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    treatments = []
    reversed_controls = []
    adapter_dir = output / "adapters"

    for reversed_objective, bucket, label in (
        (False, treatments, "treatment"),
        (True, reversed_controls, "reversed"),
    ):
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = load_model(args.model, TARGET)
            lora = install_lora(model, seed=seed)
            manifest = adapter_manifest(model, lora)
            if manifest["trainable_parameters"] != 4096:
                raise RuntimeError(
                    f"R3 trainable-parameter drift: {manifest['trainable_parameters']}"
                )

            training = train_adapter(
                model,
                lora,
                records,
                seed=seed,
                reversed_objective=reversed_objective,
                pad_id=pad_id,
            )

            reserve_scores = score_plain(
                model,
                records,
                reserve_indices,
                batch_size=2,
                pad_id=pad_id,
            )
            reserve = summarize(reserve_scores, RESERVE_ITEMS)

            row = {
                "seed": seed,
                "objective": label,
                "training": training,
                "adapter": manifest,
                "reserve": reserve,
            }

            if not reversed_objective:
                sentiment = score_control_family(model, sentiment_records)
                arithmetic = score_control_family(model, arithmetic_records)
                row["controls"] = {
                    "sentiment": sentiment,
                    "arithmetic": arithmetic,
                }

            save_adapter(
                adapter_dir / f"{label}_seed{seed}.pt",
                lora,
                {
                    "protocol": "experiments/gsd-001/WP06_R3_PROTOCOL.md",
                    "seed": seed,
                    "objective": label,
                    **manifest,
                },
            )

            bucket.append(row)
            del model, lora
            gc.collect()
            torch.cuda.empty_cache()

    treatment_states = [x["reserve"]["state"] for x in treatments]
    reversed_states = [x["reserve"]["state"] for x in reversed_controls]
    treatment_control_states = [
        {
            "sentiment": x["controls"]["sentiment"]["state"],
            "arithmetic": x["controls"]["arithmetic"]["state"],
        }
        for x in treatments
    ]

    adjudication = adjudicate_r3(
        source_state=source_reserve["state"],
        target_state=target_reserve["state"],
        treatment_states=treatment_states,
        reversed_states=reversed_states,
        treatment_control_states=treatment_control_states,
        exact_boundary_ok=exact_boundary_ok,
        base_controls_ok=base_controls_ok,
    )

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R3",
        "protocol": "experiments/gsd-001/WP06_R3_PROTOCOL.md",
        "task": TASK_KEY,
        "source_revision": SOURCE,
        "target_revision": TARGET,
        "hardware": hardware,
        "training_rows": "128-191",
        "reserve_rows": "256-286",
        "reserve_n": len(RESERVE_ITEMS),
        "exact_boundary_ok": exact_boundary_ok,
        "source_reserve": source_reserve,
        "target_reserve": target_reserve,
        "base_controls": {
            "sentiment": base_sentiment,
            "arithmetic": base_arithmetic,
            "control_ok": base_controls_ok,
        },
        "frozen_training": {
            "rank": RANK,
            "alpha": ALPHA,
            "module": "layer12.self_attn.o_proj",
            "seeds": list(SEEDS),
            "steps_per_adapter": TRAIN_STEPS,
            "batch_items": BATCH_ITEMS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "treatments": treatments,
        "reversed_controls": reversed_controls,
        **adjudication,
        "claim_boundary": (
            "R3 parameter-light recovery is ambiguous between recovery and rapid relearning"
        ),
    }
    write_json(output / "wp06_r3_result.json", result)

    source_dir = output / SOURCE
    target_dir = output / TARGET

    write_json(
        source_dir / "summary.json",
        {
            "revision": SOURCE,
            "role": "source",
            "reserve": source_reserve,
        },
    )
    write_json(
        target_dir / "summary.json",
        {
            "revision": TARGET,
            "role": "target",
            "base_reserve": target_reserve,
            "base_controls": result["base_controls"],
            "treatments": treatments,
            "reversed_controls": reversed_controls,
            "disposition": result["disposition"],
        },
    )

    common_manifest = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R3",
        "kind": "PARAMETER_LIGHT_LORA_RECOVERY",
        "protocol": "experiments/gsd-001/WP06_R3_PROTOCOL.md",
        "dataset_repository": args.dataset,
        "dataset_resolved_sha": dataset_sha,
        "task": TASK_KEY,
        "training_rows": "128-191",
        "reserve_rows": "256-286",
        "hardware": hardware,
        "precision": "bfloat16",
        "claim_boundary": (
            "parameter-light recovery only; rapid relearning not excluded"
        ),
    }
    write_json(
        source_dir / "manifest.json",
        {
            **common_manifest,
            "model_repository": args.model,
            "model_revision": SOURCE,
            "model_resolved_sha": source_sha,
            "role": "source_baseline",
        },
    )
    write_json(
        target_dir / "manifest.json",
        {
            **common_manifest,
            "model_repository": args.model,
            "model_revision": TARGET,
            "model_resolved_sha": target_sha,
            "role": "target_and_adapters",
            "adapter_module": "model.layers.11.self_attn.o_proj",
            "rank": RANK,
            "seeds": list(SEEDS),
            "treatment_count": len(treatments),
            "reversed_control_count": len(reversed_controls),
        },
    )


if __name__ == "__main__":
    main()
