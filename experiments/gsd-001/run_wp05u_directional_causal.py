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
from gsd.scoring import classify_margins, resolve_answer_start

SOURCE = "stage1-step2000-tokens5B"
FLANK = "stage1-step3000-tokens7B"
TARGET = "stage1-step4000-tokens9B"
TASK_CONFIG = "truthy_answer.surprising_truth"
TASK_KEY = "truthy_answer/surprising_truth"
DISCOVERY = list(range(0, 64))
EVALUATION = list(range(64, 128))
EXACT_MODES = {"stable_prompt_space", "forced_separate_continuation"}

CANDIDATES = {
    "target_q14h2": (14, 2),
    "control_q14h1": (14, 1),
    "control_q14h3": (14, 3),
    "control_q13h2": (13, 2),
    "control_q15h2": (15, 2),
}


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


def q_norm_module(model, layer: int):
    attn = model.model.layers[layer].self_attn
    norm = getattr(attn, "q_norm", None)
    if norm is None:
        raise RuntimeError(f"layer {layer} has no native q_norm; frozen U1 intervention surface unavailable")
    return norm


def collect_candidate_means(model, tokenizer, prompts):
    import torch
    head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    sums = {name: torch.zeros(head_dim, dtype=torch.float64) for name in CANDIDATES}
    count = 0
    for item in DISCOVERY:
        enc = tokenizer(prompts[item], return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to("cuda")
        token_pos = ids.shape[1] - 1
        captured = {}
        handles = []
        for name, (layer, head) in CANDIDATES.items():
            def make_hook(candidate_name=name, candidate_head=head):
                def hook(_module, _inputs, output):
                    vec = output[0, token_pos, candidate_head * head_dim:(candidate_head + 1) * head_dim]
                    captured[candidate_name] = vec.detach().float().cpu().double()
                    return output
                return hook
            handles.append(q_norm_module(model, layer).register_forward_hook(make_hook()))
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
        for handle in handles:
            handle.remove()
        missing = set(CANDIDATES) - set(captured)
        if missing:
            raise RuntimeError(f"missing candidate captures: {sorted(missing)}")
        for name in CANDIDATES:
            sums[name] += captured[name]
        count += 1
        del ids
    return {name: sums[name] / count for name in CANDIDATES}


def paired_bootstrap_ci(deltas, seed=0, n_boot=4000):
    import random as pyrandom
    rng = pyrandom.Random(seed)
    vals = list(map(float, deltas))
    if not vals:
        return [0.0, 0.0]
    means = []
    n = len(vals)
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]]


def score_one(model, tokenizer, prompt, answer, intervention=None):
    import torch
    ids, start, mode = resolve_answer_start(tokenizer, prompt, answer)
    inp = torch.tensor([ids], dtype=torch.long, device="cuda")
    handle = None
    if intervention is not None:
        layer, head, vector = intervention
        head_dim = vector.numel()
        token_pos = start - 1
        vector = vector.to(device="cuda", dtype=torch.bfloat16)

        def hook(_module, _inputs, output):
            patched = output.clone()
            sl = slice(head * head_dim, (head + 1) * head_dim)
            patched[0, token_pos, sl] = patched[0, token_pos, sl] + vector
            return patched

        handle = q_norm_module(model, layer).register_forward_hook(hook)

    try:
        with torch.inference_mode():
            logits = model(input_ids=inp, use_cache=False).logits[0].float()
    finally:
        if handle is not None:
            handle.remove()

    pos = torch.arange(start - 1, len(ids) - 1, device="cuda")
    tok = inp[0, start:]
    lp = torch.log_softmax(logits.index_select(0, pos), dim=-1)
    score = float(lp.gather(1, tok.unsqueeze(1)).sum().item())
    del inp, logits
    return score, mode


def score_condition(model, tokenizer, prompts, correct, incorrect, intervention=None):
    margins, modes = [], []
    for item in EVALUATION:
        c, cm = score_one(model, tokenizer, prompts[item], correct[item], intervention)
        w, wm = score_one(model, tokenizer, prompts[item], incorrect[item], intervention)
        margins.append(c - w)
        modes.extend([cm, wm])
    return margins, classify_margins(margins, n_boot=4000, seed=0), all(m in EXACT_MODES for m in modes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    import torch
    from datasets import load_dataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if torch.cuda.get_device_capability(0) < (8, 0):
        raise SystemExit("BF16-capable accelerator required")
    hardware = torch.cuda.get_device_name(0)

    dataset_sha = resolve_hf_dataset_sha(args.dataset)
    model_shas = {r: resolve_hf_model_sha(args.model, r) for r in (SOURCE, FLANK, TARGET)}
    rows = list(load_dataset(args.dataset, TASK_CONFIG, split="items", revision=dataset_sha))
    demos = [x for x in rows if x["split"] == "demo"]
    tests = [x for x in rows if x["split"] == "test"][:128]
    if len(tests) != 128:
        raise RuntimeError(f"requires 128 test rows, got {len(tests)}")
    prompts, correct, incorrect = build_base_k8_prompts(demos, tests, seed=0)

    means = {}
    for revision in (SOURCE, FLANK):
        tok = load_tokenizer(args.model, revision)
        model = load_model(args.model, revision)
        means[revision] = collect_candidate_means(model, tok, prompts)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    directions = {name: means[FLANK][name] - means[SOURCE][name] for name in CANDIDATES}

    tok = load_tokenizer(args.model, TARGET)
    model = load_model(args.model, TARGET)
    conditions = {
        "baseline": None,
        "target_reverse": (14, 2, -directions["target_q14h2"]),
        "target_amplify": (14, 2, directions["target_q14h2"]),
        "control_q14h1_reverse": (14, 1, -directions["control_q14h1"]),
        "control_q14h3_reverse": (14, 3, -directions["control_q14h3"]),
        "control_q13h2_reverse": (13, 2, -directions["control_q13h2"]),
        "control_q15h2_reverse": (15, 2, -directions["control_q15h2"]),
    }

    scored = {}
    exact_ok = True
    for name, intervention in conditions.items():
        margins, summary, exact = score_condition(model, tok, prompts, correct, incorrect, intervention)
        scored[name] = {"margins": margins, "summary": summary}
        exact_ok = exact_ok and exact

    baseline = scored["baseline"]["margins"]
    effects = {}
    for name, row in scored.items():
        if name == "baseline":
            continue
        paired = [a - b for a, b in zip(row["margins"], baseline)]
        effects[name] = {
            "mean_margin_delta": sum(paired) / len(paired),
            "paired_bootstrap_ci95": paired_bootstrap_ci(paired),
        }

    reverse = effects["target_reverse"]["mean_margin_delta"]
    amplify = effects["target_amplify"]["mean_margin_delta"]
    control_names = [x for x in effects if x.startswith("control_")]
    best_control = max(effects[x]["mean_margin_delta"] for x in control_names)

    if not exact_ok:
        disposition = "EXACT_BOUNDARY_CONTRACT_FAILED"
    elif reverse >= 0.25 and amplify <= -0.25 and reverse >= best_control + 0.10:
        disposition = "HEAD2_DIRECTIONAL_CAUSAL_SUPPORT"
    elif reverse >= 0.25 or amplify <= -0.25:
        disposition = "HEAD2_NONSPECIFIC_CAUSAL_EFFECT"
    else:
        disposition = "NO_HEAD2_CAUSAL_EFFECT"

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP05U-U1",
        "protocol": "experiments/gsd-001/WP05U_UTILITY_EXPANSION_PROTOCOL.md",
        "task": TASK_KEY,
        "hardware": hardware,
        "revisions": [SOURCE, FLANK, TARGET],
        "model_resolved_shas": model_shas,
        "dataset_resolved_sha": dataset_sha,
        "discovery_rows": "0-63",
        "evaluation_rows": "64-127",
        "fresh_holdout_claim": False,
        "cross_transition_predictive_claim": False,
        "candidate_directions": {
            name: {"layer": CANDIDATES[name][0], "head": CANDIDATES[name][1], "l2_norm": float(torch.linalg.vector_norm(directions[name].float()))}
            for name in CANDIDATES
        },
        "conditions": {k: {"summary": v["summary"]} for k, v in scored.items()},
        "paired_effects": effects,
        "exact_boundary_ok": exact_ok,
        "best_control_reverse_mean_delta": best_control,
        "disposition": disposition,
        "strongest_programme_evidence": "E2_ACCESSIBILITY_SHIFT_SUPPORTED",
        "claim_boundary": "same-transition causal intervention on previously consumed rows; no mechanism identity, persistence, or cross-transition predictive claim",
    }

    output = Path(args.output_dir)
    write_json(output / "wp05u_u1_result.json", result)
    for revision in (SOURCE, FLANK, TARGET):
        sub = output / revision
        write_json(sub / "summary.json", {
            "revision": revision,
            "disposition": disposition,
            "target_reverse_effect": effects["target_reverse"],
            "target_amplify_effect": effects["target_amplify"],
        })
        write_json(sub / "manifest.json", {
            "programme": "GSD-001",
            "work_package": "GSD-WP05U-U1",
            "model_repository": args.model,
            "model_revision": revision,
            "model_resolved_sha": model_shas[revision],
            "dataset_repository": args.dataset,
            "dataset_resolved_sha": dataset_sha,
            "hardware": hardware,
            "precision": "bfloat16",
            "protocol": "experiments/gsd-001/WP05U_UTILITY_EXPANSION_PROTOCOL.md",
            "fresh_holdout_claim": False,
        })


if __name__ == "__main__":
    main()
