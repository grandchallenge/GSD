#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

CONTENT = Path("/content")
PAYLOAD = CONTENT / "gcl_source.tar.gz"
JOB_PATH = CONTENT / "gcl_job.json"
MANIFEST_PATH = CONTENT / "gcl_manifest.json"
WORK_ROOT = CONTENT / "gsd_colab_work"
SOURCE_ROOT = WORK_ROOT / "source"
RESULT_PATH = CONTENT / "gcl_result.json"
RECEIPT_PATH = CONTENT / "experiment_receipt.json"
BUNDLE_PATH = CONTENT / "gcl_output_bundle.tar.gz"
STDOUT_PATH = CONTENT / "gsd_job_stdout.txt"
STDERR_PATH = CONTENT / "gsd_job_stderr.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"links not permitted: {member.name}")
        tf.extractall(dest, filter="data")


def verify_payload(manifest: dict) -> None:
    if sha256_file(PAYLOAD) != manifest["payload_sha256"]:
        raise RuntimeError("source payload digest mismatch")
    expected = {row["path"]: row for row in manifest["files"]}
    observed = {}
    for path in SOURCE_ROOT.rglob("*"):
        if path.is_file():
            rel = path.relative_to(SOURCE_ROOT).as_posix()
            observed[rel] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if set(observed) != set(expected):
        raise RuntimeError("source payload file-set mismatch")
    for rel, row in expected.items():
        if observed[rel] != {"bytes": row["bytes"], "sha256": row["sha256"]}:
            raise RuntimeError(f"source payload drift: {rel}")


def detect_runtime() -> dict:
    import torch
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": str(torch.version.cuda),
        "torch": str(torch.__version__),
    }


def validate_runtime(job: dict, runtime: dict) -> None:
    if not runtime["cuda_available"]:
        raise RuntimeError("requested GPU but CUDA is unavailable")
    requested = str(job["resource"]["accelerator"]).upper()
    observed = str(runtime["gpu"] or "").upper()
    aliases = {
        "T4": ("T4",),
        "L4": ("L4",),
        "A100": ("A100",),
        "H100": ("H100",),
    }
    if requested not in aliases:
        raise RuntimeError(f"unsupported GSD accelerator: {requested}")
    if not any(token in observed for token in aliases[requested]):
        raise RuntimeError(f"accelerator substitution rejected: requested={requested} observed={observed}")


def run_logged(argv: list[str], cwd: Path) -> int:
    with STDOUT_PATH.open("w", encoding="utf-8") as out, STDERR_PATH.open("w", encoding="utf-8") as err:
        proc = subprocess.run(argv, cwd=cwd, stdout=out, stderr=err, text=True, check=False)
    return proc.returncode


def make_bundle(run_root: Path | None) -> None:
    with tarfile.open(BUNDLE_PATH, "w:gz") as tf:
        for path in (RESULT_PATH, RECEIPT_PATH, STDOUT_PATH, STDERR_PATH, MANIFEST_PATH, JOB_PATH):
            if path.exists():
                tf.add(path, arcname=path.name)
        if run_root and run_root.exists():
            tf.add(run_root, arcname="runs")


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    job = {}
    manifest = {}
    runtime = {}
    status = "RED_ENGINEERING"
    returncode = 99
    fatal_error = None
    run_root = None
    try:
        job = json.loads(JOB_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        safe_extract(PAYLOAD, SOURCE_ROOT)
        verify_payload(manifest)

        runtime = detect_runtime()
        validate_runtime(job, runtime)

        source = SOURCE_ROOT / "experiments" / "gsd-001"
        upstream = SOURCE_ROOT / "external" / "GDsuite"
        run_root = CONTENT / "gsd_runs"

        install = [
            sys.executable, "-m", "pip", "install", "-q",
            "vllm>=0.6", "transformers>=4.53.2", "datasets>=3",
            "huggingface_hub>=0.25", "pyyaml>=6"
        ]
        if subprocess.run(install, check=False).returncode != 0:
            raise RuntimeError("dependency installation failed")

        revisions = list(job.get("revisions", []))
        if not revisions:
            raise RuntimeError("job must provide exact revisions")

        workload = str(job.get("workload", "gsd_wp01_checkpoint_sweep"))
        families = None

        if workload == "gsd_wp02_stable_controls":
            argv = [
                sys.executable, str(source / "run_controls_batch.py"),
                "--output-dir", str(run_root),
                "--max-eval", str(int(job.get("max_eval", 128))),
                "--sentiment-demo-k", str(int(job.get("sentiment_demo_k", 16))),
            ]
            for revision in revisions:
                argv += ["--revision", revision]
        elif workload == "gsd_wp03_truthy_adversarial":
            argv = [
                sys.executable, str(source / "run_wp03_truthy_batch.py"),
                "--output-dir", str(run_root),
                "--max-eval", str(int(job.get("max_eval", 128))),
                "--seeds", *[str(x) for x in job.get("seeds", [0, 1, 2, 3])],
                "--variants", *list(job.get("variants", [
                    "instruction_prefix", "expanded_markers", "double_newline"
                ])),
            ]
            for revision in revisions:
                argv += ["--revision", revision]
        elif workload == "gsd_wp03_truthy_boundary_repair":
            argv = [
                sys.executable, str(source / "run_wp03_truthy_batch.py"),
                "--output-dir", str(run_root),
                "--max-eval", str(int(job.get("max_eval", 128))),
                "--seeds", *[str(x) for x in job.get("seeds", [0, 1, 2, 3])],
                "--variants", *list(job.get("variants", [
                    "instruction_prefix", "expanded_markers", "double_newline"
                ])),
                "--protocol", "experiments/gsd-001/WP03_BOUNDARY_REPAIR_PROTOCOL.md",
            ]
            for revision in revisions:
                argv += ["--revision", revision]
        elif workload == "gsd_wp06_r0_context_recovery":
            argv = [
                sys.executable, str(source / "run_wp06_r0_batch.py"),
                "--output-dir", str(run_root),
                "--max-eval", str(int(job.get("max_eval", 128))),
                "--protocol", "experiments/gsd-001/WP06_R0_PROTOCOL.md",
            ]
        elif workload == "gsd_wp06_r1_activation_recovery":
            argv = [
                sys.executable, str(source / "run_wp06_r1_activation.py"),
                "--output-dir", str(run_root),
                "--max-eval", str(int(job.get("max_eval", 128))),
                "--batch-size", str(int(job.get("batch_size", 2))),
            ]
        elif workload == "gsd_wp06_r2_sparse_causal_recovery":
            argv = [
                sys.executable, str(source / "run_wp06_r2_sparse.py"),
                "--output-dir", str(run_root),
                "--batch-size", str(int(job.get("batch_size", 2))),
            ]
        elif workload == "gsd_wp06_r3_parameter_light_recovery":
            argv = [
                sys.executable, str(source / "run_wp06_r3_parameter.py"),
                "--output-dir", str(run_root),
            ]
        else:
            families = list(job.get("families", ["intuitive_answer"]))
            if not families:
                raise RuntimeError("job must provide at least one evaluation family")
            argv = [
                sys.executable, str(source / "run_sweep.py"),
                "--upstream-dir", str(upstream),
                "--output-dir", str(run_root),
                "--families", *families,
                "--n-seeds", str(int(job.get("n_seeds", 1))),
                "--max-eval", str(int(job.get("max_eval", 128))),
            ]
            for revision in revisions:
                argv += ["--revision", revision]

        returncode = run_logged(argv, SOURCE_ROOT)
        if returncode != 0:
            raise RuntimeError(f"GSD hosted workload exited with code {returncode}")

        summaries = sorted(run_root.glob("*/summary.json"))
        manifests = sorted(run_root.glob("*/manifest.json"))
        if len(summaries) != len(revisions):
            raise RuntimeError(f"summary count mismatch: {len(summaries)} != {len(revisions)}")
        if len(manifests) != len(revisions):
            raise RuntimeError(f"manifest count mismatch: {len(manifests)} != {len(revisions)}")

        RESULT_PATH.write_text(json.dumps({
            "status": "GREEN_ENGINEERING_GSD_HOSTED_SWEEP",
            "experiment_id": job["experiment_id"],
            "revision_count": len(revisions),
            "summary_count": len(summaries),
            "manifest_count": len(manifests),
            "families": families,
            "workload": workload,
            "runtime": runtime,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        status = "GREEN_ENGINEERING"
        returncode = 0
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        STDERR_PATH.write_text(
            (STDERR_PATH.read_text(encoding="utf-8") if STDERR_PATH.exists() else "")
            + "\n" + traceback.format_exc(),
            encoding="utf-8",
        )
    finally:
        receipt = {
            "schema_version": 1,
            "experiment_id": job.get("experiment_id"),
            "status": status,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "source_commit": manifest.get("source_commit"),
            "source_payload_sha256": manifest.get("payload_sha256"),
            "job_sha256": sha256_file(JOB_PATH) if JOB_PATH.exists() else None,
            "result_sha256": sha256_file(RESULT_PATH) if RESULT_PATH.exists() else None,
            "runtime": runtime,
            "workload": job.get("workload", "gsd_wp01_checkpoint_sweep"),
            "scientific_execution_authorized": True,
            "promotion_claim": False,
            "fatal_error": fatal_error,
            "returncode": returncode,
            "claim_boundary": job.get("claim_boundary"),
        }
        RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        make_bundle(run_root)
    return returncode if status == "GREEN_ENGINEERING" else max(1, returncode)


if __name__ == "__main__":
    raise SystemExit(main())
