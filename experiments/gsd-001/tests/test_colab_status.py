import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
STATUS_TOOL = ROOT / "experiments" / "gsd-001" / "colab" / "update_run_status.py"
COMMENT_TOOL = ROOT / "experiments" / "gsd-001" / "colab" / "format_run_status_comment.py"


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True)


def test_terminal_status_and_comment(tmp_path):
    status = tmp_path / "RUN_STATUS.json"
    receipt = tmp_path / "experiment_receipt.json"
    result = tmp_path / "gcl_result.json"
    comment = tmp_path / "comment.md"

    receipt.write_text(json.dumps({
        "status": "GREEN_ENGINEERING",
        "started_at": "2026-10-05T00:00:00Z",
        "finished_at": "2026-10-05T00:10:00Z",
        "runtime": {"gpu": "NVIDIA L4"},
        "fatal_error": None,
        "source_commit": "abc",
        "source_payload_sha256": "payload",
        "job_sha256": "job",
        "result_sha256": "result",
    }), encoding="utf-8")
    result.write_text(json.dumps({
        "status": "GREEN_ENGINEERING_GSD_HOSTED_SWEEP",
        "revision_count": 20,
        "summary_count": 20,
        "manifest_count": 20,
        "families": ["truthy_answer"],
        "runtime": {"gpu": "NVIDIA L4"},
    }), encoding="utf-8")

    run(STATUS_TOOL, status, "--state", "QUEUED",
        "--experiment-id", "EXP", "--run-id", "RUN",
        "--session", "SESSION", "--accelerator", "L4",
        "--source-commit", "abc", "--message", "queued")
    run(STATUS_TOOL, status, "--state", "GREEN",
        "--experiment-id", "EXP", "--run-id", "RUN",
        "--session", "SESSION", "--accelerator", "L4",
        "--source-commit", "abc", "--exit-code", "0",
        "--message", "verified", "--receipt", receipt, "--result", result)

    data = json.loads(status.read_text(encoding="utf-8"))
    assert data["state"] == "GREEN"
    assert data["terminal"] is True
    assert [x["state"] for x in data["history"]] == ["QUEUED", "GREEN"]
    assert data["result"]["summary_count"] == 20
    assert data["result"]["manifest_count"] == 20

    run(COMMENT_TOOL, status, "--output", comment)
    text = comment.read_text(encoding="utf-8")
    assert "state: `GREEN`" in text
    assert "GPU observed: `NVIDIA L4`" in text
    assert "summaries: `20`" in text
    assert "manifests: `20`" in text
