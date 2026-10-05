from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.wp03_validation import CandidateSpec, validate_candidate


CANDIDATES = [
    CandidateSpec(
        task="truthy_answer/surprising_truth",
        from_revision="stage1-step0-tokens0B",
        to_revision="stage1-step2000-tokens5B",
        from_state="PATTERN_MATCHING",
        to_state="GENERALIZING",
    ),
    CandidateSpec(
        task="truthy_answer/surprising_truth",
        from_revision="stage1-step2000-tokens5B",
        to_revision="stage1-step4000-tokens9B",
        from_state="GENERALIZING",
        to_state="PATTERN_MATCHING",
    ),
    CandidateSpec(
        task="truthy_answer/common_misconception",
        from_revision="stage1-step0-tokens0B",
        to_revision="stage1-step2000-tokens5B",
        from_state="GENERALIZING",
        to_state="PATTERN_MATCHING",
    ),
]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--revision", action="append", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["instruction_prefix", "expanded_markers", "double_newline"],
    )
    p.add_argument(
        "--pair-controls",
        default=str(HERE / "evidence" / "WP02_STABLE_CONTROLS" / "pair_controls.json"),
    )
    args = p.parse_args()

    root = Path(args.output_dir)
    worker = HERE / "run_wp03_truthy_checkpoint.py"

    for revision in args.revision:
        argv = [
            sys.executable, str(worker),
            "--revision", revision,
            "--output-dir", str(root),
            "--max-eval", str(args.max_eval),
            "--seeds", *[str(x) for x in args.seeds],
            "--variants", *args.variants,
        ]
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0:
            return proc.returncode

    summaries = {}
    for revision in args.revision:
        path = root / revision / "summary.json"
        if not path.exists():
            raise RuntimeError(f"missing WP03 summary: {path}")
        summaries[revision] = json.loads(path.read_text(encoding="utf-8"))

    pair_controls = json.loads(Path(args.pair_controls).read_text(encoding="utf-8"))
    candidate_results = []

    for spec in CANDIDATES:
        left = summaries[spec.from_revision]["tasks"][spec.task]
        right = summaries[spec.to_revision]["tasks"][spec.task]
        pair_key = f"{spec.from_revision}->{spec.to_revision}"

        variants = {
            name: (
                left["prompt_variants_seed0"][name]["state"],
                right["prompt_variants_seed0"][name]["state"],
            )
            for name in args.variants
        }
        replay = {
            int(seed): (
                left["base_replay"]["seeds"][str(seed)]["state"],
                right["base_replay"]["seeds"][str(seed)]["state"],
            )
            for seed in args.seeds
        }
        boundary_stable = bool(
            left["boundary_all_stable"] and right["boundary_all_stable"]
        )

        validation = validate_candidate(
            spec,
            control_ok=bool(pair_controls.get(pair_key, False)),
            endpoint_boundaries_stable=boundary_stable,
            variant_endpoint_states=variants,
            replay_endpoint_states=replay,
            min_replay_seeds=3,
        )

        candidate_results.append({
            "task": spec.task,
            "from_revision": spec.from_revision,
            "to_revision": spec.to_revision,
            "from_state": spec.from_state,
            "to_state": spec.to_state,
            "control_ok": bool(pair_controls.get(pair_key, False)),
            "endpoint_boundaries_stable": boundary_stable,
            "variant_endpoint_states": variants,
            "replay_endpoint_states": replay,
            **validation,
        })

    flank_states = {}
    for revision, summary in summaries.items():
        flank_states[revision] = {
            task: body["base_replay"]["pooled"]["state"]
            for task, body in summary["tasks"].items()
        }

    result = {
        "protocol": "experiments/gsd-001/WP03_PROTOCOL.md",
        "checkpoint_count": len(summaries),
        "seeds": args.seeds,
        "variants": args.variants,
        "candidate_results": candidate_results,
        "flank_states": flank_states,
        "claim_boundary": "behavioural transition validation only; no mechanism claim",
    }
    write_json(root / "wp03_result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
