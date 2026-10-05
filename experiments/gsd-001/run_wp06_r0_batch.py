from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gsd.recovery import adjudicate_r0

SOURCE = "stage1-step2000-tokens5B"
FLANK = "stage1-step3000-tokens7B"
TARGET = "stage1-step4000-tokens9B"
PAIR_KEY = f"{SOURCE}->{TARGET}"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-eval", type=int, default=128)
    p.add_argument(
        "--pair-controls",
        default=str(HERE / "evidence" / "WP02_STABLE_CONTROLS" / "pair_controls.json"),
    )
    p.add_argument(
        "--protocol",
        default="experiments/gsd-001/WP06_R0_PROTOCOL.md",
    )
    args = p.parse_args()

    output = Path(args.output_dir)
    revisions = [SOURCE, FLANK, TARGET]

    for revision in revisions:
        argv = [
            sys.executable,
            str(HERE / "run_wp06_r0_checkpoint.py"),
            "--revision", revision,
            "--output-dir", str(output),
            "--max-eval", str(args.max_eval),
        ]
        subprocess.run(argv, cwd=HERE, check=True)

    summaries = {
        revision: read_json(output / revision / "summary.json")
        for revision in revisions
    }
    controls = read_json(Path(args.pair_controls))
    control_ok = bool(controls.get(PAIR_KEY, False))

    source_states = {
        variant: data["state"]
        for variant, data in summaries[SOURCE]["variants"].items()
    }
    flank_states = {
        variant: data["state"]
        for variant, data in summaries[FLANK]["variants"].items()
    }
    target_states = {
        variant: data["state"]
        for variant, data in summaries[TARGET]["variants"].items()
    }

    exact = all(bool(s["boundary_all_exact"]) for s in summaries.values())
    adjudication = adjudicate_r0(
        source_states,
        target_states,
        control_ok=control_ok,
        exact_boundary_ok=exact,
    )

    result = {
        "programme": "GSD-001",
        "work_package": "GSD-WP06",
        "tier": "R0",
        "protocol": args.protocol,
        "source_revision": SOURCE,
        "flank_revision": FLANK,
        "target_revision": TARGET,
        "control_pair": PAIR_KEY,
        "control_ok": control_ok,
        "boundary_all_exact": exact,
        "source_states": source_states,
        "flank_states": flank_states,
        "target_states": target_states,
        **adjudication,
        "claim_boundary": "R0 accessibility only; no persistence or mechanism claim",
    }
    write_json(output / "wp06_r0_result.json", result)


if __name__ == "__main__":
    main()
