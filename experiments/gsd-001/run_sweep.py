from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.checkpoints import list_hf_checkpoint_refs, resolve_hf_dataset_sha, resolve_hf_model_sha, stride_refs
from gsd.ingest import ingest_checkpoint

EXPECTED_BLOBS = {
    "README.md": "e6a336fb540d6b47cd5cc34f495eafa8fd53a115",
    "config.yaml": "4b91818725ba025e95f4bf73397cdc8e76e5bf59",
    "run_eval.py": "cf73f27027654e5c76edb038f743fbeed5e801e8",
}


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def verify_upstream(checkout: Path) -> None:
    for name, expected in EXPECTED_BLOBS.items():
        actual = git_blob_sha1(checkout / name)
        if actual != expected:
            raise RuntimeError(f"upstream blob mismatch for {name}: {actual}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def invoke_upstream(checkout: Path, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [str(checkout / "run_eval.py"), *argv]
        runpy.run_path(str(checkout / "run_eval.py"), run_name="__main__")
    finally:
        sys.argv = old_argv


def cuda_bf16_capable(torch_mod) -> bool:
    """Use the architectural BF16 boundary required by vLLM.

    vLLM requires CUDA compute capability >= 8.0 for BF16. Some PyTorch/CUDA
    combinations may report torch.cuda.is_bf16_supported() true on older GPUs,
    so the device capability is the fail-closed authority here.
    """
    major, minor = torch_mod.cuda.get_device_capability(0)
    return (major, minor) >= (8, 0)


def derived_config(checkout: Path, run_root: Path) -> tuple[Path, str]:
    import torch
    import yaml

    if cuda_bf16_capable(torch):
        return checkout / "config.yaml", "upstream_bfloat16"

    cfg = yaml.safe_load((checkout / "config.yaml").read_text(encoding="utf-8"))
    cfg["vllm"]["dtype"] = "float16"
    path = run_root / "derived-config-fp16.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path, "declared_fp16_override_for_non_bf16_gpu"


def environment_receipt() -> dict:
    import torch
    import transformers
    import vllm

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16_supported_torch_predicate": torch.cuda.is_bf16_supported(),
        "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported_for_vllm": cuda_bf16_capable(torch),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    parser.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    parser.add_argument("--revision", action="append", default=[])
    parser.add_argument("--auto-revisions", action="store_true")
    parser.add_argument("--revision-stride", type=int, default=1)
    parser.add_argument("--families", nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--max-eval", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("GSD WP01 accelerator boundary: CUDA GPU required.")

    checkout = Path(args.upstream_dir).resolve()
    verify_upstream(checkout)
    run_root = Path(args.output_dir).resolve()
    config_path, precision_mode = derived_config(checkout, run_root)

    revisions = list(args.revision)
    if args.auto_revisions:
        refs = list_hf_checkpoint_refs(args.model)
        refs = stride_refs(refs, args.revision_stride)
        revisions.extend(r.revision for r in refs)
    revisions = list(dict.fromkeys(revisions))
    if not revisions:
        parser.error("provide --revision or --auto-revisions")

    dataset_sha = resolve_hf_dataset_sha(args.dataset)

    for revision in revisions:
        out = run_root / revision
        upstream_out = out / "upstream"
        model_sha = resolve_hf_model_sha(args.model, revision)

        argv = [
            "--model_name", args.model,
            "--revision", revision,
            "--config", str(config_path),
            "--output_dir", str(upstream_out),
            "--hf_dataset", args.dataset,
            "--n_seeds", str(args.n_seeds),
            "--max_eval", str(args.max_eval),
        ]
        if args.families:
            argv += ["--families", *args.families]
        if args.max_num_seqs is not None:
            argv += ["--max_num_seqs", str(args.max_num_seqs)]

        invoke_upstream(checkout, argv)
        rows, summary = ingest_checkpoint(upstream_out, revision)
        write_jsonl(out / "items.jsonl", rows)
        write_json(out / "summary.json", summary)
        write_json(out / "manifest.json", {
            "programme": "GSD-001",
            "tranche": "GSD-TRANCHE-A",
            "upstream_blobs": EXPECTED_BLOBS,
            "model_repository": args.model,
            "model_revision": revision,
            "model_resolved_sha": model_sha,
            "dataset_repository": args.dataset,
            "dataset_resolved_sha": dataset_sha,
            "precision_mode": precision_mode,
            "n_seeds": args.n_seeds,
            "max_eval": args.max_eval,
            "families": args.families,
            "revision_stride": args.revision_stride,
            "environment": environment_receipt(),
            "claim_boundary": "candidate behavioural evidence only; WP03 validation required",
        })


if __name__ == "__main__":
    main()
