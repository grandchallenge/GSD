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
from gsd.scoring import classify_margins, resolve_answer_start

SOURCE = "stage1-step2000-tokens5B"
FLANK = "stage1-step3000-tokens7B"
TARGET = "stage1-step4000-tokens9B"
REVISIONS = (SOURCE, FLANK, TARGET)
TASK_CONFIG = "truthy_answer.surprising_truth"
TASK_KEY = "truthy_answer/surprising_truth"
DISCOVERY = list(range(0, 64))
EVALUATION = list(range(64, 128))
EXACT_MODES = {"stable_prompt_space", "forced_separate_continuation"}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_base_k8_prompts(demos, tests, seed=0):
    rng = random.Random(seed)
    prompts, correct, incorrect = [], [], []
    for test in tests:
        chosen = rng.sample(demos, min(8, len(demos)))
        blocks = [f"{d['prompt']} {d['answer']}" for d in chosen]
        prompts.append("\n".join(blocks + [test["prompt"]]))
        correct.append(test["correct_answer"])
        incorrect.append(test["incorrect_answer"])
    return prompts, correct, incorrect


def load_tokenizer(model_repo: str, revision: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_repo, revision=revision, use_fast=True)


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


def cos_distance(a, b) -> float:
    import torch
    a = a.float()
    b = b.float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 0.0
    return float(1.0 - torch.dot(a, b) / denom)


def cos_similarity(a, b) -> float:
    import torch
    a = a.float()
    b = b.float()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def percentile_rank(values, value) -> float:
    if not values:
        return 0.0
    return sum(v <= value for v in values) / len(values)


def apply_head_norm(norm, projected, n_heads: int, head_dim: int):
    import torch
    shape = projected.shape
    x = projected.reshape(*shape[:-1], n_heads, head_dim)
    if norm is not None:
        x = norm(x)
    return x


def collect_internal_means(model, tokenizer, prompts, batch_size: int):
    import torch
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    n_q = model.config.num_attention_heads
    n_kv = getattr(model.config, "num_key_value_heads", n_q)
    head_dim = getattr(model.config, "head_dim", hidden_size // n_q)

    partitions = {"discovery": DISCOVERY, "evaluation": EVALUATION}
    sums = {}
    counts = {}
    for part in partitions:
        sums[part] = {
            "hidden": [torch.zeros(hidden_size, dtype=torch.float64) for _ in range(n_layers)],
            "q": [[torch.zeros(head_dim, dtype=torch.float64) for _ in range(n_q)] for _ in range(n_layers)],
            "k": [[torch.zeros(head_dim, dtype=torch.float64) for _ in range(n_kv)] for _ in range(n_layers)],
        }
        counts[part] = 0

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must expose pad token")

    for off in range(0, len(prompts), batch_size):
        batch_prompts = prompts[off:off + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")
        final_pos = attention_mask.sum(dim=1) - 1
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        for row in range(input_ids.shape[0]):
            item = off + row
            part = "discovery" if item < 64 else "evaluation"
            if item >= 128:
                continue
            pos = int(final_pos[row].item())
            for layer in range(n_layers):
                h = out.hidden_states[layer][row, pos].detach()
                sums[part]["hidden"][layer] += h.float().cpu().double()
                attn = model.model.layers[layer].self_attn
                with torch.inference_mode():
                    q = attn.q_proj(h)
                    k = attn.k_proj(h)
                    qn = apply_head_norm(getattr(attn, "q_norm", None), q, n_q, head_dim)
                    kn = apply_head_norm(getattr(attn, "k_norm", None), k, n_kv, head_dim)
                for head in range(n_q):
                    sums[part]["q"][layer][head] += qn[head].float().cpu().double()
                for head in range(n_kv):
                    sums[part]["k"][layer][head] += kn[head].float().cpu().double()
            counts[part] += 1
        del out, input_ids, attention_mask
        torch.cuda.empty_cache()

    result = {}
    for part in partitions:
        c = counts[part]
        result[part] = {
            "hidden": [x / c for x in sums[part]["hidden"]],
            "q": [[x / c for x in layer] for layer in sums[part]["q"]],
            "k": [[x / c for x in layer] for layer in sums[part]["k"]],
        }
    return result


def score_behaviour(model, tokenizer, prompts, correct, incorrect, items, batch_size):
    import torch
    margins = []
    modes = []
    for item in items:
        scores = {}
        for kind, answer in (("correct", correct[item]), ("incorrect", incorrect[item])):
            ids, start, mode = resolve_answer_start(tokenizer, prompts[item], answer)
            modes.append(mode)
            inp = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                logits = model(input_ids=inp, use_cache=False).logits[0].float()
            pos = torch.arange(start - 1, len(ids) - 1, device="cuda")
            tok = inp[0, start:]
            lp = torch.log_softmax(logits.index_select(0, pos), dim=-1)
            scores[kind] = float(lp.gather(1, tok.unsqueeze(1)).sum().item())
            del inp, logits
        margins.append(scores["correct"] - scores["incorrect"])
    return classify_margins(margins, n_boot=4000, seed=0), all(m in EXACT_MODES for m in modes)


def build_diagnostics(all_means):
    import torch
    out = {"representation": {}, "operators": {}}
    for part in ("discovery", "evaluation"):
        s = all_means[SOURCE][part]
        f = all_means[FLANK][part]
        t = all_means[TARGET][part]

        rep = []
        for layer in range(len(s["hidden"])):
            d23 = cos_distance(s["hidden"][layer], f["hidden"][layer])
            d34 = cos_distance(f["hidden"][layer], t["hidden"][layer])
            direction = cos_similarity(
                f["hidden"][layer] - s["hidden"][layer],
                t["hidden"][layer] - f["hidden"][layer],
            )
            rep.append({
                "layer": layer,
                "drift_2k_3k": d23,
                "drift_3k_4k": d34,
                "locality_difference": d23 - d34,
                "post_transition_direction_similarity": direction,
            })
        out["representation"][part] = rep

        ops = []
        for kind in ("q", "k"):
            for layer in range(len(s[kind])):
                for head in range(len(s[kind][layer])):
                    d23 = cos_distance(s[kind][layer][head], f[kind][layer][head])
                    d34 = cos_distance(f[kind][layer][head], t[kind][layer][head])
                    ops.append({
                        "kind": kind.upper(),
                        "layer": layer,
                        "head": head,
                        "drift_2k_3k": d23,
                        "drift_3k_4k": d34,
                        "locality_difference": d23 - d34,
                    })
        out["operators"][part] = ops
    return out


def select_and_validate(diag, means):
    disc_rep = diag["representation"]["discovery"]
    eval_rep = diag["representation"]["evaluation"]
    selected_rep = max(disc_rep, key=lambda x: x["locality_difference"])
    er = next(x for x in eval_rep if x["layer"] == selected_rep["layer"])
    rep_values = [x["locality_difference"] for x in eval_rep]
    layer = selected_rep["layer"]
    rep_dir = cos_similarity(
        means[FLANK]["evaluation"]["hidden"][layer] - means[SOURCE]["evaluation"]["hidden"][layer],
        means[FLANK]["discovery"]["hidden"][layer] - means[SOURCE]["discovery"]["hidden"][layer],
    )
    rep_eval = {
        **er,
        "evaluation_percentile": percentile_rank(rep_values, er["locality_difference"]),
        "discovery_evaluation_change_direction_cosine": rep_dir,
    }
    rep_pass = (
        er["drift_2k_3k"] >= 1.25 * max(er["drift_3k_4k"], 1e-12)
        and rep_eval["evaluation_percentile"] >= 0.75
        and rep_dir >= 0.5
    )

    disc_ops = diag["operators"]["discovery"]
    med = sorted(x["drift_2k_3k"] for x in disc_ops)[len(disc_ops)//2]
    eligible = [x for x in disc_ops if x["drift_2k_3k"] >= med]
    selected_op = max(eligible, key=lambda x: x["locality_difference"])
    eo = next(
        x for x in diag["operators"]["evaluation"]
        if x["kind"] == selected_op["kind"] and x["layer"] == selected_op["layer"] and x["head"] == selected_op["head"]
    )
    op_values = [x["locality_difference"] for x in diag["operators"]["evaluation"]]
    kind = selected_op["kind"].lower()
    layer, head = selected_op["layer"], selected_op["head"]
    op_dir = cos_similarity(
        means[FLANK]["evaluation"][kind][layer][head] - means[SOURCE]["evaluation"][kind][layer][head],
        means[FLANK]["discovery"][kind][layer][head] - means[SOURCE]["discovery"][kind][layer][head],
    )
    op_eval = {
        **eo,
        "evaluation_percentile": percentile_rank(op_values, eo["locality_difference"]),
        "discovery_evaluation_change_direction_cosine": op_dir,
    }
    op_pass = (
        eo["drift_2k_3k"] >= 1.25 * max(eo["drift_3k_4k"], 1e-12)
        and op_eval["evaluation_percentile"] >= 0.90
        and op_dir >= 0.5
    )
    return selected_rep, rep_eval, rep_pass, selected_op, op_eval, op_pass


def manifest(revision, model_repo, model_sha, dataset_repo, dataset_sha, hardware):
    return {
        "programme": "GSD-001",
        "work_package": "GSD-WP05",
        "kind": "MECHANISTIC_OPERATOR_LOCALIZATION",
        "protocol": "experiments/gsd-001/WP05_PROTOCOL.md",
        "model_repository": model_repo,
        "model_revision": revision,
        "model_resolved_sha": model_sha,
        "dataset_repository": dataset_repo,
        "dataset_resolved_sha": dataset_sha,
        "task": TASK_KEY,
        "hardware": hardware,
        "precision": "bfloat16",
        "discovery_rows": "0-63",
        "evaluation_rows": "64-127",
        "claim_boundary": "within-transition localization only; no cross-transition predictive claim",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for WP05")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required for WP05")
    hardware = torch.cuda.get_device_name(0)

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    model_shas = {r: resolve_hf_model_sha(args.model, r) for r in REVISIONS}
    rows = list(load_dataset(args.dataset, TASK_CONFIG, split="items", revision=dataset_sha))
    demos = [x for x in rows if x["split"] == "demo"]
    tests = [x for x in rows if x["split"] == "test"][:128]
    if len(tests) != 128:
        raise RuntimeError(f"WP05 requires exactly 128 test rows, got {len(tests)}")
    prompts, correct, incorrect = build_base_k8_prompts(demos, tests, seed=0)

    all_means = {}
    behaviour = {}
    exact_ok = True
    geometry = None

    for revision in REVISIONS:
        tok = load_tokenizer(args.model, revision)
        model = load_model(args.model, revision)
        current_geometry = {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
            "head_dim": getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
        }
        if geometry is None:
            geometry = current_geometry
        elif geometry != current_geometry:
            raise RuntimeError(f"checkpoint geometry drift: {current_geometry} != {geometry}")

        dsum, dexact = score_behaviour(model, tok, prompts, correct, incorrect, DISCOVERY, args.batch_size)
        esum, eexact = score_behaviour(model, tok, prompts, correct, incorrect, EVALUATION, args.batch_size)
        behaviour[revision] = {"discovery": dsum, "evaluation": esum}
        exact_ok = exact_ok and dexact and eexact
        all_means[revision] = collect_internal_means(model, tok, prompts, args.batch_size)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    diag = build_diagnostics(all_means)
    srep, erep, rep_pass, sop, eop, op_pass = select_and_validate(diag, all_means)

    baseline_ok = (
        behaviour[SOURCE]["discovery"]["state"] == "GENERALIZING"
        and behaviour[SOURCE]["evaluation"]["state"] == "GENERALIZING"
        and behaviour[FLANK]["discovery"]["state"] == "PATTERN_MATCHING"
        and behaviour[FLANK]["evaluation"]["state"] == "PATTERN_MATCHING"
        and behaviour[TARGET]["discovery"]["state"] == "PATTERN_MATCHING"
        and behaviour[TARGET]["evaluation"]["state"] == "PATTERN_MATCHING"
    )

    if not exact_ok:
        disposition = "EXACT_BOUNDARY_CONTRACT_FAILED"
    elif not baseline_ok:
        disposition = "WP05_BASELINE_UNRESOLVED"
    elif rep_pass and op_pass:
        disposition = "WP05_SIGNATURE_VALIDATED_WITHIN_TRANSITION"
    elif rep_pass or op_pass:
        disposition = "WP05_SIGNATURE_PARTIAL"
    else:
        disposition = "WP05_SIGNATURE_REJECTED"

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP05",
        "protocol": "experiments/gsd-001/WP05_PROTOCOL.md",
        "task": TASK_KEY,
        "revisions": list(REVISIONS),
        "hardware": hardware,
        "geometry": geometry,
        "behaviour": behaviour,
        "exact_boundary_ok": exact_ok,
        "baseline_ok": baseline_ok,
        "selected_representation_discovery": srep,
        "selected_representation_evaluation": erep,
        "representation_pass": rep_pass,
        "selected_operator_discovery": sop,
        "selected_operator_evaluation": eop,
        "operator_pass": op_pass,
        "disposition": disposition,
        "strongest_programme_evidence": "E2_ACCESSIBILITY_SHIFT_SUPPORTED",
        "cross_transition_predictive_claim": False,
        "claim_boundary": "WP05 validates or rejects a within-transition signature only; one confirmed transition is insufficient for cross-transition prediction.",
        "diagnostics": diag,
    }

    output = Path(args.output_dir)
    write_json(output / "wp05_result.json", result)
    for revision in REVISIONS:
        sub = output / revision
        write_json(sub / "summary.json", {
            "revision": revision,
            "behaviour": behaviour[revision],
            "selected_representation_layer": srep["layer"],
            "selected_operator": {"kind": sop["kind"], "layer": sop["layer"], "head": sop["head"]},
            "disposition": disposition,
        })
        write_json(sub / "manifest.json", manifest(
            revision, args.model, model_shas[revision], args.dataset, dataset_sha, hardware
        ))


if __name__ == "__main__":
    main()
