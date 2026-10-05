#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("status_file", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    s = json.loads(args.status_file.read_text(encoding="utf-8"))
    receipt = s.get("receipt") or {}
    result = s.get("result") or {}
    runtime = receipt.get("runtime") or result.get("runtime") or {}

    lines = [
        "GSD hosted-run terminal receipt",
        "",
        f"- experiment: `{s.get('experiment_id')}`",
        f"- run: `{s.get('run_id')}`",
        f"- state: `{s.get('state')}`",
        f"- accelerator requested: `{s.get('accelerator')}`",
        f"- GPU observed: `{runtime.get('gpu')}`",
        f"- source commit: `{s.get('source_commit')}`",
        f"- exit code: `{s.get('exit_code')}`",
    ]
    if result:
        lines += [
            f"- revisions: `{result.get('revision_count')}`",
            f"- summaries: `{result.get('summary_count')}`",
        ]
        if result.get("manifest_count") is not None:
            lines.append(f"- manifests: `{result.get('manifest_count')}`")
        if result.get("families") is not None:
            lines.append(f"- families: `{result.get('families')}`")
    if s.get("message"):
        lines.append(f"- verification: {s.get('message')}")
    if receipt.get("fatal_error"):
        lines.append(f"- fatal error: `{receipt.get('fatal_error')}`")
    lines += [
        "",
        "Terminal state was emitted only after receipt/bundle verification and Colab session cleanup.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
