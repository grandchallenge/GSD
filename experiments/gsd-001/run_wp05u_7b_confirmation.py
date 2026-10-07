from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.wp03_validation import CandidateSpec, validate_candidate

TASK = "truthy_answer/surprising_truth"
LEFT = "stage1-step565000"
RIGHT = "stage1-step705000"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/Olmo-3-1025-7B")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument(
        "--variants",
        nargs="+",
        default=["instruction_prefix", "expanded_markers", "double_newline"],
    )
    args = p.parse_args()

    root = Path(args.output_dir)
    worker = HERE / "run_wp03_truthy_checkpoint.py"

    for revision in (LEFT, RIGHT):
        argv = [
            sys.executable, str(worker),
            "--model", args.model,
            "--revision", revision,
            "--output-dir", str(root),
            "--max-eval", str(args.max_eval),
            "--seeds", *[str(x) for x in args.seeds],
            "--variants", *args.variants,
        ]
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0:
            return proc.returncode

    left_summary = json.loads((root / LEFT / "summary.json").read_text(encoding="utf-8"))
    right_summary = json.loads((root / RIGHT / "summary.json").read_text(encoding="utf-8"))

    left = left_summary["tasks"][TASK]
    right = right_summary["tasks"][TASK]

    variant_states = {
        name: (
            left["prompt_variants_seed0"][name]["state"],
            right["prompt_variants_seed0"][name]["state"],
        )
        for name in args.variants
    }
    replay_states = {
        int(seed): (
            left["base_replay"]["seeds"][str(seed)]["state"],
            right["base_replay"]["seeds"][str(seed)]["state"],
        )
        for seed in args.seeds
    }
    boundary_stable = bool(
        left["boundary_all_exact"] and right["boundary_all_exact"]
    )

    spec = CandidateSpec(
        task=TASK,
        from_revision=LEFT,
        to_revision=RIGHT,
        from_state="PATTERN_MATCHING",
        to_state="GENERALIZING",
    )
    validation = validate_candidate(
        spec,
        control_ok=True,
        endpoint_boundaries_stable=boundary_stable,
        variant_endpoint_states=variant_states,
        replay_endpoint_states=replay_states,
        min_replay_seeds=3,
    )

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP05U-U2C",
        "protocol": "experiments/gsd-001/WP05U_7B_CONFIRM_ADVERSARIAL_PROTOCOL.md",
        "model": args.model,
        "task": TASK,
        "from_revision": LEFT,
        "to_revision": RIGHT,
        "control_receipt": "experiments/gsd-001/receipts/WP05U_U2B_OLMO3_7B_CONTROLS_A100_RECEIPT.md",
        "control_ok": True,
        "endpoint_boundaries_exact": boundary_stable,
        "variant_endpoint_states": variant_states,
        "replay_endpoint_states": replay_states,
        "seeds": args.seeds,
        "variants": args.variants,
        **validation,
        "claim_boundary": "behavioural transition confirmation only; no transfer/mechanism claim",
    }
    write_json(root / "wp05u_u2c_result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
