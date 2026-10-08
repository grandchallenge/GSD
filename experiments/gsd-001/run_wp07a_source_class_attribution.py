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
from run_wp05_mechanistic import (
    build_base_k8_prompts,
    cos_similarity,
    load_model,
    load_tokenizer,
    score_behaviour,
)

SOURCE = "stage1-step2000-tokens5B"
FLANK = "stage1-step3000-tokens7B"
MODEL = "allenai/OLMo-2-0425-1B-early-training"
EVAL_DATASET = "jiaxin-wen/generalization-dynamics-evals"
TRAIN_DATASET = "allenai/olmo-mix-1124"
TASK_CONFIG = "truthy_answer.surprising_truth"
TASK_KEY = "truthy_answer/surprising_truth"
DISCOVERY = list(range(0, 64))
CLASSES = [
    "algebraic-stack",
    "arxiv",
    "dclm",
    "open-web-math",
    "pes2o",
    "starcoder",
    "wiki",
]
SEQ_LEN = 4096
MICROBATCH = 2
GRAD_ACCUM = 4
OPT_STEPS = 8
TRAIN_TOKENS = SEQ_LEN * MICROBATCH * GRAD_ACCUM * OPT_STEPS
STREAM_SEED = 1729
LR = 2.4e-4


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def targeted_means(model, tokenizer, prompts, batch_size: int = 4):
    import torch

    hidden_sum = None
    q_sum = None
    count = 0
    layer_hidden = 13
    layer_q = 14
    head = 2

    for off in range(0, len(prompts), batch_size):
        batch = prompts[off : off + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to("cuda")
        mask = enc["attention_mask"].to("cuda")
        final_pos = mask.sum(dim=1) - 1

        captured_hidden = []
        captured_q = []

        def hidden_hook(_module, inputs):
            h = inputs[0]
            captured_hidden.append(h)

        def q_pre_hook(module, inputs):
            h = inputs[0]
            q = module.q_proj(h)
            q_norm = getattr(module, "q_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            n_heads = model.config.num_attention_heads
            head_dim = getattr(model.config, "head_dim", model.config.hidden_size // n_heads)
            q = q.reshape(*q.shape[:-1], n_heads, head_dim)
            captured_q.append(q)

        h_handle = model.model.layers[layer_hidden].register_forward_pre_hook(hidden_hook)
        q_handle = model.model.layers[layer_q].self_attn.register_forward_pre_hook(q_pre_hook)
        try:
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            h_handle.remove()
            q_handle.remove()

        h = captured_hidden[0]
        q = captured_q[0]
        for row in range(ids.shape[0]):
            pos = int(final_pos[row].item())
            hv = h[row, pos].detach().float().cpu().double()
            qv = q[row, pos, head].detach().float().cpu().double()
            hidden_sum = hv if hidden_sum is None else hidden_sum + hv
            q_sum = qv if q_sum is None else q_sum + qv
            count += 1

        del ids, mask, captured_hidden, captured_q
        torch.cuda.empty_cache()

    return {
        "layer13_hidden": hidden_sum / count,
        "q14h2": q_sum / count,
        "count": count,
    }


def collect_training_sequences(tokenizer, class_name: str, *, seed: int = STREAM_SEED):
    import torch
    from datasets import load_dataset

    ds = load_dataset(
        TRAIN_DATASET,
        class_name,
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=2048)

    eos = tokenizer.eos_token_id
    token_buffer: list[int] = []
    sequences = []
    need_sequences = MICROBATCH * GRAD_ACCUM * OPT_STEPS

    for row in ds:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ids = list(ids) + [eos]
        token_buffer.extend(ids)

        while len(token_buffer) >= SEQ_LEN and len(sequences) < need_sequences:
            chunk = token_buffer[:SEQ_LEN]
            del token_buffer[:SEQ_LEN]
            sequences.append(torch.tensor(chunk, dtype=torch.long))

        if len(sequences) >= need_sequences:
            break

    if len(sequences) != need_sequences:
        raise RuntimeError(
            f"{class_name}: insufficient streamed tokens, got {len(sequences)} sequences "
            f"of required {need_sequences}"
        )
    return torch.stack(sequences)


def make_optimizer(model):
    import torch

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embed_tokens" in name or "wte" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def train_microcontinuation(model, sequences, *, seed: int):
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    model.config.use_cache = False
    opt = make_optimizer(model)
    losses = []
    index = 0

    for step in range(OPT_STEPS):
        opt.zero_grad(set_to_none=True)
        step_losses = []
        for _ in range(GRAD_ACCUM):
            batch = sequences[index : index + MICROBATCH].to("cuda")
            index += MICROBATCH
            out = model(input_ids=batch, labels=batch, use_cache=False)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            step_losses.append(float(out.loss.detach().float().cpu()))
            del out, batch
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(sum(step_losses) / len(step_losses))

    model.eval()
    del opt
    torch.cuda.empty_cache()
    return losses


def serialise_means(means):
    return {
        "layer13_hidden_norm": float(means["layer13_hidden"].float().norm()),
        "q14h2_norm": float(means["q14h2"].float().norm()),
        "count": means["count"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP07A")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP07A")

    hardware = torch.cuda.get_device_name(0)
    output = Path(args.output_dir)

    eval_sha = resolve_hf_dataset_sha(EVAL_DATASET)
    model_shas = {
        SOURCE: resolve_hf_model_sha(MODEL, SOURCE),
        FLANK: resolve_hf_model_sha(MODEL, FLANK),
    }

    rows = list(load_dataset(EVAL_DATASET, TASK_CONFIG, split="items", revision=eval_sha))
    demos = [x for x in rows if x["split"] == "demo"]
    tests = [x for x in rows if x["split"] == "test"][:128]
    prompts, correct, incorrect = build_base_k8_prompts(demos, tests, seed=0)
    discovery_prompts = [prompts[i] for i in DISCOVERY]

    tok_source = load_tokenizer(MODEL, SOURCE)

    source_model = load_model(MODEL, SOURCE)
    source_behaviour, source_exact = score_behaviour(
        source_model, tok_source, prompts, correct, incorrect, DISCOVERY, args.batch_size
    )
    source_means = targeted_means(source_model, tok_source, discovery_prompts, args.batch_size)
    del source_model
    gc.collect()
    torch.cuda.empty_cache()

    tok_flank = load_tokenizer(MODEL, FLANK)
    flank_model = load_model(MODEL, FLANK)
    flank_behaviour, flank_exact = score_behaviour(
        flank_model, tok_flank, prompts, correct, incorrect, DISCOVERY, args.batch_size
    )
    flank_means = targeted_means(flank_model, tok_flank, discovery_prompts, args.batch_size)
    del flank_model
    gc.collect()
    torch.cuda.empty_cache()

    if not source_exact or not flank_exact:
        raise RuntimeError("WP07A exact answer-boundary contract failed")
    if source_behaviour["state"] != "GENERALIZING":
        raise RuntimeError(f"WP07A source state drifted: {source_behaviour['state']}")
    if flank_behaviour["state"] != "PATTERN_MATCHING":
        raise RuntimeError(f"WP07A flank state drifted: {flank_behaviour['state']}")

    original_hidden_delta = flank_means["layer13_hidden"] - source_means["layer13_hidden"]
    original_q_delta = flank_means["q14h2"] - source_means["q14h2"]

    class_results = []
    for class_index, class_name in enumerate(CLASSES):
        sequences = collect_training_sequences(tok_source, class_name, seed=STREAM_SEED)

        model = load_model(MODEL, SOURCE)
        losses = train_microcontinuation(model, sequences, seed=STREAM_SEED + class_index)
        post_behaviour, exact = score_behaviour(
            model, tok_source, prompts, correct, incorrect, DISCOVERY, args.batch_size
        )
        post_means = targeted_means(model, tok_source, discovery_prompts, args.batch_size)
        del model, sequences
        gc.collect()
        torch.cuda.empty_cache()

        if not exact:
            raise RuntimeError(f"{class_name}: exact boundary contract failed after continuation")

        margin_delta = float(post_behaviour["mean_margin"] - source_behaviour["mean_margin"])
        hidden_delta = post_means["layer13_hidden"] - source_means["layer13_hidden"]
        q_delta = post_means["q14h2"] - source_means["q14h2"]
        hidden_alignment = cos_similarity(hidden_delta, original_hidden_delta)
        q_alignment = cos_similarity(q_delta, original_q_delta)
        toward = -margin_delta
        eligible = toward > 0 and hidden_alignment > 0 and q_alignment > 0

        class_results.append(
            {
                "class": class_name,
                "training_tokens": TRAIN_TOKENS,
                "optimizer_steps": OPT_STEPS,
                "losses": losses,
                "source_mean_margin": source_behaviour["mean_margin"],
                "post_mean_margin": post_behaviour["mean_margin"],
                "post_state": post_behaviour["state"],
                "behaviour_margin_delta": margin_delta,
                "toward_transition_score": toward,
                "layer13_transition_direction_cosine": hidden_alignment,
                "q14h2_transition_direction_cosine": q_alignment,
                "eligible": eligible,
                "post_internal_summary": serialise_means(post_means),
            }
        )

    eligible = [x for x in class_results if x["eligible"]]
    if eligible:
        selected = sorted(
            eligible,
            key=lambda x: (
                x["toward_transition_score"],
                x["q14h2_transition_direction_cosine"],
                x["layer13_transition_direction_cosine"],
            ),
            reverse=True,
        )[0]
        disposition = "WP07A_CLASS_ATTRIBUTION_CANDIDATE"
        selected_class = selected["class"]
    else:
        disposition = "WP07A_NO_CLASS_ATTRIBUTION_CANDIDATE"
        selected_class = None

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP07A",
        "protocol": "experiments/gsd-001/WP07A_SOURCE_CLASS_ATTRIBUTION_PROTOCOL.md",
        "hardware": hardware,
        "model": MODEL,
        "source_revision": SOURCE,
        "flank_revision": FLANK,
        "model_resolved_shas": model_shas,
        "evaluation_dataset": EVAL_DATASET,
        "evaluation_dataset_sha": eval_sha,
        "training_dataset": TRAIN_DATASET,
        "training_classes": CLASSES,
        "discovery_rows": "0-63",
        "training_tokens_per_class": TRAIN_TOKENS,
        "baseline_source_behaviour": source_behaviour,
        "flank_behaviour": flank_behaviour,
        "source_internal_summary": serialise_means(source_means),
        "flank_internal_summary": serialise_means(flank_means),
        "class_results": class_results,
        "selected_class": selected_class,
        "disposition": disposition,
        "claim_boundary": (
            "source-class attribution discovery only; exact original data-window identity, "
            "held-out attribution, and causal mechanism identity are not claimed"
        ),
    }
    write_json(output / "wp07a_result.json", result)

    summary_dir = output / SOURCE
    write_json(
        summary_dir / "summary.json",
        {
            "revision": SOURCE,
            "disposition": disposition,
            "selected_class": selected_class,
            "class_count": len(class_results),
            "training_tokens_per_class": TRAIN_TOKENS,
        },
    )
    write_json(
        summary_dir / "manifest.json",
        {
            "programme": "GSD-001",
            "work_package": "GSD-WP07A",
            "protocol": "experiments/gsd-001/WP07A_SOURCE_CLASS_ATTRIBUTION_PROTOCOL.md",
            "model_repository": MODEL,
            "model_revision": SOURCE,
            "model_resolved_sha": model_shas[SOURCE],
            "flank_revision": FLANK,
            "flank_resolved_sha": model_shas[FLANK],
            "evaluation_dataset": EVAL_DATASET,
            "evaluation_dataset_sha": eval_sha,
            "training_dataset": TRAIN_DATASET,
            "hardware": hardware,
            "precision": "bfloat16",
            "claim_boundary": "discovery pilot only; no WP07 exit claim",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
