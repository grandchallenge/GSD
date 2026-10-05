from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.activation_recovery import PATCH_POINTS, adjudicate_r1, choose_discovery_layer
from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.scoring import classify_margins, resolve_answer_start
from run_wp06_r0_checkpoint import build_context_prompts

SOURCE = "stage1-step2000-tokens5B"
FLANK = "stage1-step3000-tokens7B"
TARGET = "stage1-step4000-tokens9B"
TASK_CONFIG = "truthy_answer.surprising_truth"
TASK_KEY = "truthy_answer/surprising_truth"
PAIR_KEY = f"{SOURCE}->{TARGET}"
DISCOVERY_ITEMS = list(range(0, 64))
HELDOUT_ITEMS = list(range(64, 128))
EXACT_MODES = {"stable_prompt_space", "forced_separate_continuation"}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model(model_repo: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        revision=revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    return model


def load_tokenizer(model_repo: str, revision: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_repo, revision=revision, use_fast=True)


def make_records(tokenizer, prompts, correct, incorrect):
    records = []
    modes = []
    for item, (prompt, ca, ia) in enumerate(zip(prompts, correct, incorrect)):
        for kind, answer in (("correct", ca), ("incorrect", ia)):
            ids, start, mode = resolve_answer_start(tokenizer, prompt, answer)
            if start <= 0 or start >= len(ids):
                raise RuntimeError(f"invalid answer boundary item={item} kind={kind}")
            pred_positions = list(range(start - 1, len(ids) - 1))
            if not pred_positions:
                raise RuntimeError(f"empty prediction span item={item} kind={kind}")
            records.append({
                "index": len(records),
                "item": item,
                "kind": kind,
                "ids": list(ids),
                "start": start,
                "pred_positions": pred_positions,
                "boundary_mode": mode,
            })
            modes.append(mode)
    return records, modes


def record_indices(items: list[int]) -> list[int]:
    wanted = set(items)
    return [2 * i + k for i in items for k in (0, 1) if i in wanted]


def collate(records, indices, pad_id):
    import torch
    max_len = max(len(records[i]["ids"]) for i in indices)
    input_ids = torch.full((len(indices), max_len), int(pad_id), dtype=torch.long, device="cuda")
    attention_mask = torch.zeros((len(indices), max_len), dtype=torch.long, device="cuda")
    for row, idx in enumerate(indices):
        ids = records[idx]["ids"]
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device="cuda")
        attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


def scores_from_logits(logits, input_ids, records, indices):
    import torch
    out = {}
    for row, idx in enumerate(indices):
        positions = records[idx]["pred_positions"]
        pos = torch.tensor(positions, dtype=torch.long, device=logits.device)
        selected = logits[row].index_select(0, pos).float()
        next_tokens = input_ids[row].index_select(0, pos + 1)
        logp = torch.log_softmax(selected, dim=-1)
        token_logp = logp.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
        out[idx] = float(token_logp.sum().item())
    return out


def margins_for_items(scores, items):
    return [scores[2 * i] - scores[2 * i + 1] for i in items]


def summarize(scores, items):
    return classify_margins(margins_for_items(scores, items), n_boot=4000, seed=0)


def capture_module(model, point: int):
    if point < model.config.num_hidden_layers:
        return model.model.layers[point - 1]
    if point == model.config.num_hidden_layers:
        return model.model.layers[-1]
    raise ValueError(f"unsupported patch point {point}")


def patch_module(model, point: int):
    if point < model.config.num_hidden_layers:
        return model.model.layers[point]
    if point == model.config.num_hidden_layers:
        return model.model.norm
    raise ValueError(f"unsupported patch point {point}")


def score_and_capture(model, records, indices, *, patch_points, batch_size, pad_id):
    import torch
    caches = {p: {} for p in patch_points}
    scores = {}
    captured = {}

    handles = []
    for point in patch_points:
        module = capture_module(model, point)
        def hook(_module, _args, output, point=point):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[point] = tensor.detach()
        handles.append(module.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            for off in range(0, len(indices), batch_size):
                batch = indices[off: off + batch_size]
                input_ids, attention_mask = collate(records, batch, pad_id)
                captured.clear()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                scores.update(scores_from_logits(outputs.logits, input_ids, records, batch))
                for point in patch_points:
                    h = captured[point]
                    for row, idx in enumerate(batch):
                        pos = torch.tensor(records[idx]["pred_positions"], dtype=torch.long, device=h.device)
                        caches[point][idx] = h[row].index_select(0, pos).to("cpu")
                del outputs, input_ids, attention_mask
    finally:
        for handle in handles:
            handle.remove()
    return scores, caches


def score_plain(model, records, indices, *, batch_size, pad_id):
    import torch
    scores = {}
    with torch.inference_mode():
        for off in range(0, len(indices), batch_size):
            batch = indices[off: off + batch_size]
            input_ids, attention_mask = collate(records, batch, pad_id)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            scores.update(scores_from_logits(outputs.logits, input_ids, records, batch))
            del outputs, input_ids, attention_mask
    return scores


def score_patched(model, records, indices, *, point, cache, batch_size, pad_id):
    import torch
    scores = {}
    context = {"batch": []}

    def pre_hook(_module, args):
        hidden = args[0].clone()
        for row, idx in enumerate(context["batch"]):
            pos = torch.tensor(records[idx]["pred_positions"], dtype=torch.long, device=hidden.device)
            patch = cache[idx].to(device=hidden.device, dtype=hidden.dtype)
            if patch.shape[0] != pos.numel() or patch.shape[-1] != hidden.shape[-1]:
                raise RuntimeError(f"patch shape mismatch idx={idx}")
            hidden[row, pos, :] = patch
        return (hidden, *args[1:])

    handle = patch_module(model, point).register_forward_pre_hook(pre_hook)
    try:
        with torch.inference_mode():
            for off in range(0, len(indices), batch_size):
                batch = indices[off: off + batch_size]
                context["batch"] = batch
                input_ids, attention_mask = collate(records, batch, pad_id)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                scores.update(scores_from_logits(outputs.logits, input_ids, records, batch))
                del outputs, input_ids, attention_mask
    finally:
        handle.remove()
    return scores


def verify_tokenizer(records, tokenizer, prompts, correct, incorrect):
    for item, (prompt, ca, ia) in enumerate(zip(prompts, correct, incorrect)):
        for kind, answer, idx in (
            ("correct", ca, 2 * item),
            ("incorrect", ia, 2 * item + 1),
        ):
            ids, start, mode = resolve_answer_start(tokenizer, prompt, answer)
            rec = records[idx]
            if list(ids) != rec["ids"] or start != rec["start"] or mode != rec["boundary_mode"]:
                raise RuntimeError(f"tokenizer drift at item={item} kind={kind}")


def shuffled_cache(records, heldout_indices, selected_cache, seed=17):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx in heldout_indices:
        rec = records[idx]
        groups[(rec["kind"], len(rec["pred_positions"]))].append(idx)
    mapping = {}
    for members in groups.values():
        ordered = list(members)
        rng.shuffle(ordered)
        if len(ordered) > 1:
            rotated = ordered[1:] + ordered[:1]
        else:
            rotated = ordered
        for dst, src in zip(ordered, rotated):
            mapping[dst] = selected_cache[src]
    return mapping


def max_margin_drift(a_scores, b_scores, items):
    a = margins_for_items(a_scores, items)
    b = margins_for_items(b_scores, items)
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def manifest_for(revision, model_repo, model_sha, dataset_repo, dataset_sha, result):
    return {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R1",
        "kind": "CROSS_CHECKPOINT_RESIDUAL_PATCH",
        "protocol": "experiments/gsd-001/WP06_R1_PROTOCOL.md",
        "model_repository": model_repo,
        "model_revision": revision,
        "model_resolved_sha": model_sha,
        "dataset_repository": dataset_repo,
        "dataset_resolved_sha": dataset_sha,
        "task": TASK_KEY,
        "patch_points": list(PATCH_POINTS),
        "selected_layer": result.get("selected_layer"),
        "hardware": result.get("hardware"),
        "precision": "bfloat16",
        "claim_boundary": "R1 activation recovery; no literal circuit identity claim",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--pair-controls",
        default=str(HERE / "evidence" / "WP02_STABLE_CONTROLS" / "pair_controls.json"),
    )
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP06-R1")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP06-R1")
    hardware = torch.cuda.get_device_name(0)

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    model_shas = {
        r: resolve_hf_model_sha(args.model, r)
        for r in (SOURCE, FLANK, TARGET)
    }

    rows = list(load_dataset(args.dataset, TASK_CONFIG, split="items", revision=dataset_sha))
    demos = [x for x in rows if x["split"] == "demo"]
    tests = [x for x in rows if x["split"] == "test"][:args.max_eval]
    if len(tests) != 128:
        raise RuntimeError(f"WP06-R1 requires exactly 128 test rows, got {len(tests)}")

    source_tokenizer = load_tokenizer(args.model, SOURCE)
    prompts, correct, incorrect = build_context_prompts(
        demos, tests, variant="base_k8", seed=0
    )
    records, boundary_modes = make_records(source_tokenizer, prompts, correct, incorrect)
    exact_boundary_ok = all(mode in EXACT_MODES for mode in boundary_modes)
    all_indices = list(range(len(records)))
    discovery_indices = record_indices(DISCOVERY_ITEMS)
    heldout_indices = record_indices(HELDOUT_ITEMS)
    pad_id = source_tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must expose a pad token")

    source_model = load_model(args.model, SOURCE)
    if source_model.config.num_hidden_layers != 16 or source_model.config.hidden_size != 2048:
        raise RuntimeError("OLMo2 geometry drifted from frozen WP06-R1 protocol")
    source_scores, source_cache = score_and_capture(
        source_model,
        records,
        all_indices,
        patch_points=PATCH_POINTS,
        batch_size=args.batch_size,
        pad_id=pad_id,
    )
    source_discovery = summarize(source_scores, DISCOVERY_ITEMS)
    source_heldout = summarize(source_scores, HELDOUT_ITEMS)
    del source_model
    gc.collect()
    torch.cuda.empty_cache()

    target_tokenizer = load_tokenizer(args.model, TARGET)
    verify_tokenizer(records, target_tokenizer, prompts, correct, incorrect)
    target_model = load_model(args.model, TARGET)

    target_base_scores = score_plain(
        target_model, records, all_indices, batch_size=args.batch_size, pad_id=pad_id
    )
    target_discovery = summarize(target_base_scores, DISCOVERY_ITEMS)
    target_heldout = summarize(target_base_scores, HELDOUT_ITEMS)

    discovery_patch = {}
    for point in PATCH_POINTS:
        scores = score_patched(
            target_model,
            records,
            discovery_indices,
            point=point,
            cache=source_cache[point],
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        discovery_patch[point] = {
            "summary": summarize(scores, DISCOVERY_ITEMS),
        }

    selected_layer = choose_discovery_layer(
        {p: discovery_patch[p]["summary"]["state"] for p in PATCH_POINTS},
        source_state=source_discovery["state"],
        target_state=target_discovery["state"],
        exact_boundary_ok=exact_boundary_ok,
    )

    matched_target_summary = None
    identity_summary = None
    identity_drift = None
    shuffled_summary = None
    target_self_cache = None

    if selected_layer is not None:
        matched_scores = score_patched(
            target_model,
            records,
            heldout_indices,
            point=selected_layer,
            cache=source_cache[selected_layer],
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        matched_target_summary = summarize(matched_scores, HELDOUT_ITEMS)

        base_capture_scores, self_caches = score_and_capture(
            target_model,
            records,
            heldout_indices,
            patch_points=(selected_layer,),
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        target_self_cache = self_caches[selected_layer]
        identity_scores = score_patched(
            target_model,
            records,
            heldout_indices,
            point=selected_layer,
            cache=target_self_cache,
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        identity_summary = summarize(identity_scores, HELDOUT_ITEMS)
        identity_drift = max_margin_drift(
            base_capture_scores, identity_scores, HELDOUT_ITEMS
        )

        shuffled = shuffled_cache(
            records,
            heldout_indices,
            source_cache[selected_layer],
            seed=17,
        )
        shuffled_scores = score_patched(
            target_model,
            records,
            heldout_indices,
            point=selected_layer,
            cache=shuffled,
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        shuffled_summary = summarize(shuffled_scores, HELDOUT_ITEMS)

    del target_model
    gc.collect()
    torch.cuda.empty_cache()

    flank_tokenizer = load_tokenizer(args.model, FLANK)
    verify_tokenizer(records, flank_tokenizer, prompts, correct, incorrect)
    flank_model = load_model(args.model, FLANK)
    flank_base_scores = score_plain(
        flank_model, records, heldout_indices, batch_size=args.batch_size, pad_id=pad_id
    )
    flank_heldout = summarize(flank_base_scores, HELDOUT_ITEMS)
    matched_flank_summary = None
    if selected_layer is not None:
        matched_flank_scores = score_patched(
            flank_model,
            records,
            heldout_indices,
            point=selected_layer,
            cache=source_cache[selected_layer],
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        matched_flank_summary = summarize(matched_flank_scores, HELDOUT_ITEMS)
    del flank_model
    gc.collect()
    torch.cuda.empty_cache()

    controls = json.loads(Path(args.pair_controls).read_text(encoding="utf-8"))
    control_ok = bool(controls.get(PAIR_KEY, False))

    adjudication = adjudicate_r1(
        selected_layer=selected_layer,
        source_heldout_state=source_heldout["state"],
        target_heldout_state=target_heldout["state"],
        matched_target_state=(
            matched_target_summary["state"] if matched_target_summary else None
        ),
        flank_heldout_state=flank_heldout["state"],
        matched_flank_state=(
            matched_flank_summary["state"] if matched_flank_summary else None
        ),
        identity_state=identity_summary["state"] if identity_summary else None,
        identity_max_margin_drift=identity_drift,
        shuffled_state=shuffled_summary["state"] if shuffled_summary else None,
        control_ok=control_ok,
        exact_boundary_ok=exact_boundary_ok,
    )

    mean_pred_positions = sum(
        len(r["pred_positions"]) for r in records
    ) / max(1, len(records))
    cost = None
    if selected_layer is not None:
        cost = {
            "source_layers_imported": selected_layer,
            "mean_residual_vectors_patched_per_sequence": mean_pred_positions,
            "hidden_dimensions": 2048,
        }

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R1",
        "protocol": "experiments/gsd-001/WP06_R1_PROTOCOL.md",
        "task": TASK_KEY,
        "source_revision": SOURCE,
        "flank_revision": FLANK,
        "target_revision": TARGET,
        "hardware": hardware,
        "control_ok": control_ok,
        "boundary_all_exact": exact_boundary_ok,
        "patch_points": list(PATCH_POINTS),
        "source_discovery": source_discovery,
        "source_heldout": source_heldout,
        "target_discovery": target_discovery,
        "target_heldout": target_heldout,
        "discovery_patch_states": {
            str(p): discovery_patch[p]["summary"] for p in PATCH_POINTS
        },
        "selected_layer": selected_layer,
        "selected_cost": cost,
        "matched_target_heldout": matched_target_summary,
        "flank_heldout": flank_heldout,
        "matched_flank_heldout": matched_flank_summary,
        "identity_heldout": identity_summary,
        "identity_max_margin_drift": identity_drift,
        "shuffled_heldout": shuffled_summary,
        **adjudication,
        "claim_boundary": (
            "R1 activation recovery only; literal circuit identity remains unproven"
        ),
    }

    output = Path(args.output_dir)
    write_json(output / "wp06_r1_result.json", result)

    per_revision = {
        SOURCE: {
            "revision": SOURCE,
            "role": "source",
            "discovery": source_discovery,
            "heldout": source_heldout,
        },
        TARGET: {
            "revision": TARGET,
            "role": "primary_target",
            "discovery": target_discovery,
            "heldout": target_heldout,
            "matched_heldout": matched_target_summary,
            "identity_heldout": identity_summary,
            "shuffled_heldout": shuffled_summary,
        },
        FLANK: {
            "revision": FLANK,
            "role": "localization_flank",
            "heldout": flank_heldout,
            "matched_heldout": matched_flank_summary,
        },
    }
    for revision, summary in per_revision.items():
        out = output / revision
        write_json(out / "summary.json", summary)
        write_json(
            out / "manifest.json",
            manifest_for(
                revision,
                args.model,
                model_shas[revision],
                args.dataset,
                dataset_sha,
                result,
            ),
        )


if __name__ == "__main__":
    main()
