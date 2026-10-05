#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_quiet(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def session_active(colab: str, auth: str, session: str) -> bool:
    proc = run_quiet([colab, f"--auth={auth}", "sessions"])
    return session in (proc.stdout + proc.stderr)


def extract_result(bundle: Path, output: Path) -> None:
    if output.exists() or not bundle.exists():
        return
    with tarfile.open(bundle, "r:gz") as tf:
        member = next((m for m in tf.getmembers() if Path(m.name).name == "gcl_result.json"), None)
        if member is None:
            return
        src = tf.extractfile(member)
        if src is None:
            return
        output.write_bytes(src.read())


def verify(run_dir: Path) -> tuple[bool, str]:
    receipt_path = run_dir / "experiment_receipt.json"
    manifest_path = run_dir / "gcl_manifest.json"
    job_path = run_dir / "gcl_job.json"
    result_path = run_dir / "gcl_result.json"
    bundle_path = run_dir / "gcl_output_bundle.tar.gz"

    if not receipt_path.exists():
        return False, "missing experiment_receipt.json"
    if not bundle_path.exists():
        return False, "missing gcl_output_bundle.tar.gz"
    extract_result(bundle_path, result_path)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "GREEN_ENGINEERING":
        return False, f"receipt status={receipt.get('status')}"
    if not result_path.exists():
        return False, "missing gcl_result.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = len(job.get("revisions", []))

    if receipt.get("source_commit") != manifest.get("source_commit"):
        return False, "source commit mismatch"
    if receipt.get("source_payload_sha256") != manifest.get("payload_sha256"):
        return False, "source payload digest mismatch"
    if receipt.get("job_sha256") != sha256(job_path):
        return False, "job digest mismatch"
    if receipt.get("result_sha256") != sha256(result_path):
        return False, "result digest mismatch"
    if int(result.get("revision_count", -1)) != expected:
        return False, f"revision_count mismatch: {result.get('revision_count')} != {expected}"
    if int(result.get("summary_count", -1)) != expected:
        return False, f"summary_count mismatch: {result.get('summary_count')} != {expected}"

    manifest_count = result.get("manifest_count")
    if manifest_count is None:
        with tarfile.open(bundle_path, "r:gz") as tf:
            manifest_count = sum(
                1 for m in tf.getmembers()
                if m.isfile() and m.name.endswith("/manifest.json") and m.name.startswith("runs/")
            )
    if int(manifest_count) != expected:
        return False, f"manifest_count mismatch: {manifest_count} != {expected}"
    return True, "receipt, digests, summaries, and manifests verified"


def call_status(tool: Path, status_file: Path, state: str, args, message: str, exit_code: int) -> None:
    cmd = [
        args.python, str(tool), str(status_file), "--state", state,
        "--experiment-id", args.experiment_id,
        "--run-id", args.run_dir.name,
        "--session", args.session,
        "--accelerator", args.accelerator,
        "--source-commit", args.source_commit,
        "--exit-code", str(exit_code),
        "--message", message,
    ]
    receipt = args.run_dir / "experiment_receipt.json"
    result = args.run_dir / "gcl_result.json"
    if receipt.exists():
        cmd += ["--receipt", str(receipt)]
    if result.exists():
        cmd += ["--result", str(result)]
    subprocess.run(cmd, check=True)


def notify(comment_tool: Path, status_file: Path, args) -> None:
    marker = args.run_dir / ".github-terminal-notified"
    if marker.exists():
        return
    comment = args.run_dir / "github-terminal-comment.md"
    subprocess.run([args.python, str(comment_tool), str(status_file), "--output", str(comment)], check=True)
    proc = subprocess.run([
        args.gh, "issue", "comment", str(args.issue),
        "--repo", args.repo, "--body-file", str(comment)
    ], check=False)
    if proc.returncode == 0:
        marker.touch()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--accelerator", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--colab", default=str(Path.home() / ".local/bin/colab"))
    p.add_argument("--auth", default="oauth2")
    p.add_argument("--python", default="/usr/bin/python3")
    p.add_argument("--gh", default="/usr/bin/gh")
    p.add_argument("--repo", default="grandchallenge/GSD")
    p.add_argument("--issue", type=int, default=1)
    p.add_argument("--poll-seconds", type=int, default=15)
    p.add_argument("--post-session-grace-seconds", type=int, default=90)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[3]
    status_tool = root / "experiments/gsd-001/colab/update_run_status.py"
    comment_tool = root / "experiments/gsd-001/colab/format_run_status_comment.py"
    status_file = args.run_dir / "RUN_STATUS.json"

    call_status(status_tool, status_file, "RUNNING", args, "Existing hosted run adopted by recovery watcher.", 0)

    while session_active(args.colab, args.auth, args.session):
        time.sleep(args.poll_seconds)

    deadline = time.time() + args.post_session_grace_seconds
    while time.time() < deadline:
        if (args.run_dir / "experiment_receipt.json").exists() and (args.run_dir / "gcl_output_bundle.tar.gz").exists():
            break
        time.sleep(5)

    ok, message = verify(args.run_dir)
    state = "GREEN" if ok else "RED"
    rc = 0 if ok else 1
    call_status(status_tool, status_file, state, args, message, rc)
    notify(comment_tool, status_file, args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
