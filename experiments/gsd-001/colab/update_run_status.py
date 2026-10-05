#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("status_file", type=Path)
    p.add_argument("--state", required=True,
                   choices=["QUEUED", "ALLOCATING", "RUNNING", "VERIFYING", "GREEN", "RED"])
    p.add_argument("--experiment-id")
    p.add_argument("--run-id")
    p.add_argument("--session")
    p.add_argument("--accelerator")
    p.add_argument("--source-commit")
    p.add_argument("--exit-code", type=int)
    p.add_argument("--message")
    p.add_argument("--receipt", type=Path)
    p.add_argument("--result", type=Path)
    args = p.parse_args()

    current = read_json(args.status_file)
    history = list(current.get("history", []))
    now = datetime.now(timezone.utc).isoformat()
    history.append({"state": args.state, "at": now, "message": args.message})

    out = {
        **current,
        "schema_version": 1,
        "programme": "GSD-001",
        "tranche": "GSD-TRANCHE-A",
        "state": args.state,
        "updated_at": now,
        "terminal": args.state in {"GREEN", "RED"},
        "history": history,
    }
    for key, val in {
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "session": args.session,
        "accelerator": args.accelerator,
        "source_commit": args.source_commit,
        "exit_code": args.exit_code,
        "message": args.message,
    }.items():
        if val is not None:
            out[key] = val

    if args.receipt and args.receipt.exists():
        receipt = read_json(args.receipt)
        out["receipt"] = {
            "status": receipt.get("status"),
            "started_at": receipt.get("started_at"),
            "finished_at": receipt.get("finished_at"),
            "runtime": receipt.get("runtime"),
            "fatal_error": receipt.get("fatal_error"),
            "source_commit": receipt.get("source_commit"),
            "source_payload_sha256": receipt.get("source_payload_sha256"),
            "job_sha256": receipt.get("job_sha256"),
            "result_sha256": receipt.get("result_sha256"),
        }

    if args.result and args.result.exists():
        result = read_json(args.result)
        out["result"] = {
            "status": result.get("status"),
            "revision_count": result.get("revision_count"),
            "summary_count": result.get("summary_count"),
            "manifest_count": result.get("manifest_count"),
            "families": result.get("families"),
            "runtime": result.get("runtime"),
        }

    write_atomic(args.status_file, out)
    print(json.dumps({"state": out["state"], "updated_at": out["updated_at"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
