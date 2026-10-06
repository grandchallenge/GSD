from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import resolve_hf_dataset_sha, resolve_hf_model_sha
from gsd.sparse_recovery import CANDIDATES, adjudicate_r2, candidate_key, choose_sparse_candidate
from run_wp06_r1_activation import (
    EXACT_MODES,
    FLANK,
    PAIR_KEY,
    SOURCE,
    TARGET,
    TASK_CONFIG,
    TASK_KEY,
    build_base_k8_prompts,
    collate,
    load_model,
    load_tokenizer,
    make_records,
    max_margin_drift,
    record_indices,
    score_plain,
    scores_from_logits,
    shuffled_cache,
    summarize,
    verify_tokenizer,
    write_json,
)

DISCOVERY_ITEMS = list(range(128, 192))
HELDOUT_ITEMS = list(range(192, 256))
RESERVE_ITEMS = list(range(256, 287))


def module_for(model, layer: int, kind: str):
    block = model.model.layers[layer - 1]
    if kind == "attention":
        return block.post_attention_layernorm
    if kind == "mlp":
        return block.post_feedforward_layernorm
    raise ValueError(f"unknown module kind: {kind}")


def score_and_capture_modules(
    model,
    records,
    indices,
    *,
    candidates,
    batch_size,
    pad_id,
):
    import torch

    keys = [candidate_key(layer, kind) for layer, kind in candidates]
    caches = {key: {} for key in keys}
    captured = {}
    scores = {}
    handles = []

    for layer, kind in candidates:
        key = candidate_key(layer, kind)
        module = module_for(model, layer, kind)

        def hook(_module, _args, output, key=key):
            captured[key] = output.detach()

        handles.append(module.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            for off in range(0, len(indices), batch_size):
                batch = indices[off: off + batch_size]
                input_ids, attention_mask = collate(records, batch, pad_id)
                captured.clear()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                scores.update(
                    scores_from_logits(outputs.logits, input_ids, records, batch)
                )
                for key in keys:
                    h = captured[key]
                    for row, idx in enumerate(batch):
                        pos = torch.tensor(
                            records[idx]["pred_positions"],
                            dtype=torch.long,
                            device=h.device,
                        )
                        caches[key][idx] = h[row].index_select(0, pos).to("cpu")
                del outputs, input_ids, attention_mask
    finally:
        for handle in handles:
            handle.remove()

    return scores, caches


def score_module_patched(
    model,
    records,
    indices,
    *,
    layer,
    kind,
    cache,
    batch_size,
    pad_id,
):
    import torch

    scores = {}
    context = {"batch": []}

    def hook(_module, _args, output):
        patched = output.clone()
        for row, idx in enumerate(context["batch"]):
            pos = torch.tensor(
                records[idx]["pred_positions"],
                dtype=torch.long,
                device=patched.device,
            )
            source = cache[idx].to(device=patched.device, dtype=patched.dtype)
            if source.shape[0] != pos.numel() or source.shape[-1] != patched.shape[-1]:
                raise RuntimeError(f"sparse patch shape mismatch idx={idx}")
            patched[row, pos, :] = source
        return patched

    handle = module_for(model, layer, kind).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            for off in range(0, len(indices), batch_size):
                batch = indices[off: off + batch_size]
                context["batch"] = batch
                input_ids, attention_mask = collate(records, batch, pad_id)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                scores.update(
                    scores_from_logits(outputs.logits, input_ids, records, batch)
                )
                del outputs, input_ids, attention_mask
    finally:
        handle.remove()

    return scores


def parse_candidate(key: str) -> tuple[int, str]:
    left, kind = key.split("_", 1)
    return int(left.removeprefix("layer")), kind


def manifest_for(
    revision: str,
    model_repo: str,
    model_sha: str,
    dataset_repo: str,
    dataset_sha: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R2",
        "kind": "SPARSE_CAUSAL_MODULE_PATCH",
        "protocol": "experiments/gsd-001/WP06_R2_PROTOCOL.md",
        "model_repository": model_repo,
        "model_revision": revision,
        "model_resolved_sha": model_sha,
        "dataset_repository": dataset_repo,
        "dataset_resolved_sha": dataset_sha,
        "task": TASK_KEY,
        "candidate_modules": [
            candidate_key(layer, kind) for layer, kind in CANDIDATES
        ],
        "selected_candidate": result.get("selected_candidate"),
        "hardware": result.get("hardware"),
        "precision": "bfloat16",
        "fresh_discovery_rows": "128-191",
        "fresh_heldout_rows": "192-255",
        "reserved_rows": "256-286",
        "claim_boundary": "R2 sparse causal recovery; literal circuit identity unproven",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--pair-controls",
        default=str(
            HERE / "evidence" / "WP02_STABLE_CONTROLS" / "pair_controls.json"
        ),
    )
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP06-R2")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP06-R2")
    hardware = torch.cuda.get_device_name(0)

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    model_shas = {
        revision: resolve_hf_model_sha(args.model, revision)
        for revision in (SOURCE, FLANK, TARGET)
    }

    rows = list(
        load_dataset(
            args.dataset,
            TASK_CONFIG,
            split="items",
            revision=dataset_sha,
        )
    )
    demos = [x for x in rows if x["split"] == "demo"]
    all_tests = [x for x in rows if x["split"] == "test"]
    if len(all_tests) < 287:
        raise RuntimeError(
            f"WP06-R2 requires 287 test rows, observed {len(all_tests)}"
        )

    tests = all_tests[:256]
    source_tokenizer = load_tokenizer(args.model, SOURCE)
    prompts, correct, incorrect = build_base_k8_prompts(
        demos, tests, seed=0
    )
    records, boundary_modes = make_records(
        source_tokenizer, prompts, correct, incorrect
    )
    exact_boundary_ok = all(mode in EXACT_MODES for mode in boundary_modes)
    discovery_indices = record_indices(DISCOVERY_ITEMS)
    heldout_indices = record_indices(HELDOUT_ITEMS)
    r2_indices = discovery_indices + heldout_indices
    pad_id = source_tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must expose a pad token")

    source_model = load_model(args.model, SOURCE)
    if (
        source_model.config.num_hidden_layers != 16
        or source_model.config.hidden_size != 2048
    ):
        raise RuntimeError("OLMo2 geometry drifted from frozen WP06-R2 protocol")

    source_scores, source_caches = score_and_capture_modules(
        source_model,
        records,
        r2_indices,
        candidates=CANDIDATES,
        batch_size=args.batch_size,
        pad_id=pad_id,
    )
    source_discovery = summarize(source_scores, DISCOVERY_ITEMS)
    source_heldout = summarize(source_scores, HELDOUT_ITEMS)
    del source_model
    gc.collect()
    torch.cuda.empty_cache()

    target_tokenizer = load_tokenizer(args.model, TARGET)
    verify_tokenizer(
        records, target_tokenizer, prompts, correct, incorrect
    )
    target_model = load_model(args.model, TARGET)

    target_base_scores = score_plain(
        target_model,
        records,
        r2_indices,
        batch_size=args.batch_size,
        pad_id=pad_id,
    )
    target_discovery = summarize(target_base_scores, DISCOVERY_ITEMS)
    target_heldout = summarize(target_base_scores, HELDOUT_ITEMS)

    discovery_patch = {}
    for layer, kind in CANDIDATES:
        key = candidate_key(layer, kind)
        scores = score_module_patched(
            target_model,
            records,
            discovery_indices,
            layer=layer,
            kind=kind,
            cache=source_caches[key],
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        discovery_patch[key] = summarize(scores, DISCOVERY_ITEMS)

    selected_candidate = choose_sparse_candidate(
        {
            key: summary["state"]
            for key, summary in discovery_patch.items()
        },
        source_state=source_discovery["state"],
        target_state=target_discovery["state"],
        exact_boundary_ok=exact_boundary_ok,
    )

    matched_target_summary = None
    identity_summary = None
    identity_drift = None
    shuffled_summary = None

    if selected_candidate is not None:
        selected_layer, selected_kind = parse_candidate(selected_candidate)

        matched_scores = score_module_patched(
            target_model,
            records,
            heldout_indices,
            layer=selected_layer,
            kind=selected_kind,
            cache=source_caches[selected_candidate],
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        matched_target_summary = summarize(
            matched_scores, HELDOUT_ITEMS
        )

        base_capture_scores, self_caches = score_and_capture_modules(
            target_model,
            records,
            heldout_indices,
            candidates=((selected_layer, selected_kind),),
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        self_cache = self_caches[selected_candidate]
        identity_scores = score_module_patched(
            target_model,
            records,
            heldout_indices,
            layer=selected_layer,
            kind=selected_kind,
            cache=self_cache,
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        identity_summary = summarize(identity_scores, HELDOUT_ITEMS)
        identity_drift = max_margin_drift(
            base_capture_scores,
            identity_scores,
            HELDOUT_ITEMS,
        )

        shuffled = shuffled_cache(
            records,
            heldout_indices,
            source_caches[selected_candidate],
            seed=17,
        )
        shuffled_scores = score_module_patched(
            target_model,
            records,
            heldout_indices,
            layer=selected_layer,
            kind=selected_kind,
            cache=shuffled,
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        shuffled_summary = summarize(
            shuffled_scores, HELDOUT_ITEMS
        )

    del target_model
    gc.collect()
    torch.cuda.empty_cache()

    flank_tokenizer = load_tokenizer(args.model, FLANK)
    verify_tokenizer(
        records, flank_tokenizer, prompts, correct, incorrect
    )
    flank_model = load_model(args.model, FLANK)
    flank_base_scores = score_plain(
        flank_model,
        records,
        heldout_indices,
        batch_size=args.batch_size,
        pad_id=pad_id,
    )
    flank_heldout = summarize(flank_base_scores, HELDOUT_ITEMS)
    matched_flank_summary = None

    if selected_candidate is not None:
        selected_layer, selected_kind = parse_candidate(selected_candidate)
        matched_flank_scores = score_module_patched(
            flank_model,
            records,
            heldout_indices,
            layer=selected_layer,
            kind=selected_kind,
            cache=source_caches[selected_candidate],
            batch_size=args.batch_size,
            pad_id=pad_id,
        )
        matched_flank_summary = summarize(
            matched_flank_scores, HELDOUT_ITEMS
        )

    del flank_model
    gc.collect()
    torch.cuda.empty_cache()

    controls = json.loads(
        Path(args.pair_controls).read_text(encoding="utf-8")
    )
    control_ok = bool(controls.get(PAIR_KEY, False))

    adjudication = adjudicate_r2(
        selected_candidate=selected_candidate,
        source_heldout_state=source_heldout["state"],
        target_heldout_state=target_heldout["state"],
        matched_target_state=(
            matched_target_summary["state"]
            if matched_target_summary
            else None
        ),
        flank_heldout_state=flank_heldout["state"],
        matched_flank_state=(
            matched_flank_summary["state"]
            if matched_flank_summary
            else None
        ),
        identity_state=(
            identity_summary["state"] if identity_summary else None
        ),
        identity_max_margin_drift=identity_drift,
        shuffled_state=(
            shuffled_summary["state"] if shuffled_summary else None
        ),
        control_ok=control_ok,
        exact_boundary_ok=exact_boundary_ok,
    )

    selected_cost = None
    if selected_candidate is not None:
        selected_cost = {
            "modules_patched": 1,
            "mean_residual_vectors_patched_per_sequence": (
                sum(len(records[i]["pred_positions"]) for i in r2_indices)
                / len(r2_indices)
            ),
            "hidden_dimensions": 2048,
        }

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R2",
        "protocol": "experiments/gsd-001/WP06_R2_PROTOCOL.md",
        "task": TASK_KEY,
        "source_revision": SOURCE,
        "flank_revision": FLANK,
        "target_revision": TARGET,
        "hardware": hardware,
        "fresh_test_count": len(all_tests),
        "discovery_rows": "128-191",
        "heldout_rows": "192-255",
        "reserved_rows": "256-286",
        "control_ok": control_ok,
        "boundary_all_exact": exact_boundary_ok,
        "candidate_modules": [
            candidate_key(layer, kind) for layer, kind in CANDIDATES
        ],
        "source_discovery": source_discovery,
        "source_heldout": source_heldout,
        "target_discovery": target_discovery,
        "target_heldout": target_heldout,
        "discovery_patch_states": discovery_patch,
        "selected_candidate": selected_candidate,
        "selected_cost": selected_cost,
        "matched_target_heldout": matched_target_summary,
        "flank_heldout": flank_heldout,
        "matched_flank_heldout": matched_flank_summary,
        "identity_heldout": identity_summary,
        "identity_max_margin_drift": identity_drift,
        "shuffled_heldout": shuffled_summary,
        **adjudication,
        "claim_boundary": (
            "R2 sparse causal recovery only; literal circuit identity unproven"
        ),
    }

    output = Path(args.output_dir)
    write_json(output / "wp06_r2_result.json", result)

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
