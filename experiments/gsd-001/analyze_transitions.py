from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.transitions import (
    CheckpointSummary,
    detect_candidate_transitions,
    detect_state_reversals,
)


def load_control_map(path: str | None) -> dict[str, bool]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("control JSON must map revision names to booleans")
    return {str(k): bool(v) for k, v in data.items()}


def load_pair_control_map(path: str | None) -> dict[str, bool]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("pair-control JSON must map from->to revision keys to booleans")
    return {str(k): bool(v) for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--task", help="Exact task key, e.g. flipped_answer/sst2")
    target.add_argument("--family", help="Descriptive family aggregate")
    parser.add_argument(
        "--control-json",
        default=None,
        help="Legacy JSON object mapping exact revisions to stable-control pass/fail.",
    )
    parser.add_argument(
        "--pair-control-json",
        default=None,
        help="JSON object mapping exact from_revision->to_revision pairs to stable-control pass/fail.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    control_map = load_control_map(args.control_json)
    pair_control_map = load_pair_control_map(args.pair_control_json)
    root = Path(args.run_root)
    summaries: list[CheckpointSummary] = []
    section = "tasks" if args.task else "families"
    key = args.task or args.family

    for path in sorted(root.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        selected = data.get(section, {}).get(key)
        if selected is None:
            continue
        revision = data["revision"]
        summaries.append(
            CheckpointSummary(
                revision=revision,
                step=int(data["step"]),
                state=selected["state"],
                mean_margin=float(selected["mean_margin"]),
                ci_low=float(selected["ci95"][0]),
                ci_high=float(selected["ci95"][1]),
                n=int(selected["n"]),
                control_ok=control_map.get(revision, False),
            )
        )

    reversals = detect_state_reversals(summaries)
    if pair_control_map:
        for reversal in reversals:
            pair_key = f"{reversal['from_revision']}->{reversal['to_revision']}"
            reversal["control_validated"] = bool(pair_control_map.get(pair_key, False))
            if reversal["control_validated"]:
                reversal["status"] = "CANDIDATE_TRANSITION"
        candidates = [dict(r) for r in reversals if r["control_validated"]]
    else:
        candidates = detect_candidate_transitions(summaries)
    result = {
        "target_type": "task" if args.task else "family_aggregate",
        "target": key,
        "checkpoint_count": len(summaries),
        "control_evidence_provided": (
            args.control_json is not None or args.pair_control_json is not None
        ),
        "control_evidence_mode": (
            "pair" if args.pair_control_json is not None
            else "revision" if args.control_json is not None
            else "none"
        ),
        "state_reversals": reversals,
        "candidate_transitions": candidates,
        "claim_boundary": (
            "A state reversal is not a candidate transition until its checkpoint "
            "pair has explicit stable-control evidence. WP03 validation is still "
            "required after candidate promotion."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
