from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.control_validation import paired_control_disposition


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--revision", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument("--sentiment-demo-k", type=int, default=16)
    p.add_argument("--model", default="allenai/OLMo-2-0425-1B-early-training")
    p.add_argument("--dataset", default="jiaxin-wen/generalization-dynamics-evals")
    args = p.parse_args()

    root = Path(args.output_dir)
    worker = HERE / "run_controls.py"

    for revision in args.revision:
        argv = [
            sys.executable, str(worker),
            "--model", args.model,
            "--dataset", args.dataset,
            "--revision", revision,
            "--output-dir", str(root),
            "--max-eval", str(args.max_eval),
            "--sentiment-demo-k", str(args.sentiment_demo_k),
        ]
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0:
            return proc.returncode

    summaries = []
    for revision in args.revision:
        path = root / revision / "summary.json"
        if not path.exists():
            raise RuntimeError(f"missing control summary after isolated worker: {path}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    summaries.sort(key=lambda x: int(x["step"]))
    pair_map = {}
    pair_evidence = {}
    for left, right in zip(summaries, summaries[1:]):
        key = f"{left['revision']}->{right['revision']}"
        disposition = paired_control_disposition(
            left["controls"], right["controls"], n_boot=4000, seed=0
        )
        pair_map[key] = bool(disposition["control_ok"])
        pair_evidence[key] = disposition

    write_json(root / "pair_controls.json", pair_map)
    write_json(root / "pair_control_evidence.json", pair_evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
