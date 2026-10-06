from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.continued_recovery import adjudicate_r4
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
from run_wp06_r3_control_repair import (
    build_control_prompts,
    delta_evidence,
    score_controls,
)

TRAIN_ITEMS = list(range(128, 192))
EVAL_ITEMS = list(range(192, 256))
SEEDS = (0, 1, 2)
EVAL_STEPS = (0, 1, 2, 4, 8, 16, 32)
MAX_STEPS = 32
BATCH_ITEMS = 2
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0


def answer_nll(model, records, indices, pad_id: int):
    import torch

    input_ids, attention_mask = collate(records, indices, pad_id)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    losses = []
    token_count = 0
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
        losses.append(-token_logp.sum())
        token_count += len(positions)
    return torch.stack(losses).sum() / max(1, token_count), token_count


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_truthy(model, records, pad_id: int):
    scores = score_plain(
        model,
        records,
        record_indices(EVAL_ITEMS),
        batch_size=2,
        pad_id=pad_id,
    )
    return summarize(scores, EVAL_ITEMS)


def run_lane(
    *,
    model_repo: str,
    target_revision: str,
    records,
    pad_id: int,
    seed: int,
    lane: str,
    base_truthy: dict,
    base_controls: dict,
    tokenizer,
    sentiment_records,
    arithmetic_records,
    lm_texts,
):
    import torch

    if lane not in {"GENERALIZATION_SELECTED", "PATTERN_SELECTED"}:
        raise ValueError(f"unknown R4 lane: {lane}")

    torch.manual_seed(seed)
    model = load_model(model_repo, target_revision)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()

    trainable = trainable_parameter_count(model)
    total = sum(p.numel() for p in model.parameters())
    if trainable != total:
        raise RuntimeError("R4 requires all target-model parameters to be trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )

    order = list(TRAIN_ITEMS)
    random.Random(seed).shuffle(order)

    trajectory = [{
        "step": 0,
        "answer_tokens_processed": 0,
        "summary": base_truthy,
    }]
    recovery_step = None
    recovery_cost = None
    recovery_controls = None
    cumulative_tokens = 0
    losses = []

    for step in range(1, MAX_STEPS + 1):
        batch_items = order[(step - 1) * BATCH_ITEMS : step * BATCH_ITEMS]
        if len(batch_items) != BATCH_ITEMS:
            raise RuntimeError("R4 one-pass batch schedule drifted")

        if lane == "GENERALIZATION_SELECTED":
            indices = [2 * item for item in batch_items]
        else:
            indices = [2 * item + 1 for item in batch_items]

        optimizer.zero_grad(set_to_none=True)
        model.train()
        loss, token_count = answer_nll(model, records, indices, pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        cumulative_tokens += int(token_count)

        if step in EVAL_STEPS:
            model.eval()
            summary = evaluate_truthy(model, records, pad_id)
            trajectory.append({
                "step": step,
                "answer_tokens_processed": cumulative_tokens,
                "summary": summary,
            })

            if (
                lane == "GENERALIZATION_SELECTED"
                and recovery_step is None
                and summary["state"] == "GENERALIZING"
            ):
                recovery_step = step
                adapted_controls = score_controls(
                    model,
                    tokenizer,
                    sentiment_records,
                    arithmetic_records,
                    lm_texts,
                    pad_id,
                )
                recovery_controls = {
                    "raw": adapted_controls,
                    "delta_vs_base": delta_evidence(base_controls, adapted_controls),
                }
                recovery_cost = {
                    "trainable_parameters": trainable,
                    "optimizer_steps": step,
                    "answer_tokens_processed": cumulative_tokens,
                }

            model.train()

    out = {
        "seed": seed,
        "lane": lane,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP,
        },
        "trajectory": trajectory,
        "recovery_step": recovery_step,
        "recovery_cost": recovery_cost,
        "recovery_controls": recovery_controls,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "pattern_generalizing": any(
            row["summary"]["state"] == "GENERALIZING"
            for row in trajectory
        ) if lane == "PATTERN_SELECTED" else False,
    }

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP06-R4")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP06-R4")
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
        raise RuntimeError(f"R4 frozen contract requires 287 test rows, got {len(tests)}")

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

    sentiment_triplet, arithmetic_triplet, lm_texts = build_control_prompts(
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
    exact_boundary_ok = all(
        mode in EXACT_MODES
        for mode in truth_modes + sentiment_modes + arithmetic_modes
    )

    source_model = load_model(args.model, SOURCE)
    source_eval = evaluate_truthy(source_model, records, pad_id)
    del source_model
    gc.collect()
    torch.cuda.empty_cache()

    target_tokenizer = load_tokenizer(args.model, TARGET)
    verify_tokenizer(records, target_tokenizer, prompts, correct, incorrect)
    verify_tokenizer(sentiment_records, target_tokenizer, *sentiment_triplet)
    verify_tokenizer(arithmetic_records, target_tokenizer, *arithmetic_triplet)

    target_model = load_model(args.model, TARGET)
    target_eval = evaluate_truthy(target_model, records, pad_id)
    base_controls = score_controls(
        target_model,
        target_tokenizer,
        sentiment_records,
        arithmetic_records,
        lm_texts,
        pad_id,
    )
    del target_model
    gc.collect()
    torch.cuda.empty_cache()

    treatments = []
    pattern_controls = []
    for seed in SEEDS:
        treatments.append(
            run_lane(
                model_repo=args.model,
                target_revision=TARGET,
                records=records,
                pad_id=pad_id,
                seed=seed,
                lane="GENERALIZATION_SELECTED",
                base_truthy=target_eval,
                base_controls=base_controls,
                tokenizer=target_tokenizer,
                sentiment_records=sentiment_records,
                arithmetic_records=arithmetic_records,
                lm_texts=lm_texts,
            )
        )
        pattern_controls.append(
            run_lane(
                model_repo=args.model,
                target_revision=TARGET,
                records=records,
                pad_id=pad_id,
                seed=seed,
                lane="PATTERN_SELECTED",
                base_truthy=target_eval,
                base_controls=base_controls,
                tokenizer=target_tokenizer,
                sentiment_records=sentiment_records,
                arithmetic_records=arithmetic_records,
                lm_texts=lm_texts,
            )
        )

    treatment_steps = [row["recovery_step"] for row in treatments]
    treatment_control_ok = [
        (
            row["recovery_controls"]["delta_vs_base"]["control_ok"]
            if row["recovery_controls"] is not None
            else None
        )
        for row in treatments
    ]
    pattern_generalizing = [row["pattern_generalizing"] for row in pattern_controls]

    adjudication = adjudicate_r4(
        source_state=source_eval["state"],
        target_state=target_eval["state"],
        treatment_recovery_steps=treatment_steps,
        treatment_control_ok=treatment_control_ok,
        pattern_generalizing=pattern_generalizing,
        exact_boundary_ok=exact_boundary_ok,
    )

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R4",
        "protocol": "experiments/gsd-001/WP06_R4_PROTOCOL.md",
        "task": TASK_KEY,
        "source_revision": SOURCE,
        "target_revision": TARGET,
        "hardware": hardware,
        "training_rows": "128-191",
        "evaluation_rows": "192-255",
        "fresh_holdout_claim": False,
        "all_surprising_truth_rows_previously_consumed": True,
        "exact_boundary_ok": exact_boundary_ok,
        "source_eval": source_eval,
        "target_eval": target_eval,
        "base_controls": base_controls,
        "frozen_training": {
            "seeds": list(SEEDS),
            "max_steps": MAX_STEPS,
            "eval_steps": list(EVAL_STEPS),
            "batch_items": BATCH_ITEMS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP,
            "trainable_scope": "all model parameters",
            "loss": "answer-token causal NLL",
        },
        "treatments": treatments,
        "pattern_controls": pattern_controls,
        **adjudication,
        "claim_boundary": (
            "R4 measures continued-training recovery cost; persistence and mechanism change remain unresolved"
        ),
    }

    root = Path(args.output_dir)
    write_json(root / "wp06_r4_result.json", result)

    for revision, role, summary, sha in (
        (SOURCE, "source_reference", source_eval, source_sha),
        (TARGET, "target_continuation", target_eval, target_sha),
    ):
        out = root / revision
        write_json(out / "summary.json", {
            "revision": revision,
            "role": role,
            "baseline": summary,
            "disposition": result["disposition"],
        })
        write_json(out / "manifest.json", {
            "programme": "GSD-001",
            "work_package": "GSD-WP06",
            "tier": "R4",
            "kind": "FULL_MODEL_SELECTED_DATA_CONTINUATION",
            "protocol": "experiments/gsd-001/WP06_R4_PROTOCOL.md",
            "model_repository": args.model,
            "model_revision": revision,
            "model_resolved_sha": sha,
            "dataset_repository": args.dataset,
            "dataset_resolved_sha": dataset_sha,
            "training_rows": "128-191",
            "evaluation_rows": "192-255",
            "fresh_holdout_claim": False,
            "hardware": hardware,
            "precision": "bfloat16",
            "claim_boundary": (
                "continued-training recovery cost only; no E3/E4 claim"
            ),
        })


if __name__ == "__main__":
    main()
