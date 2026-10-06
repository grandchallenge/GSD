from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_7b_transfer_pilot_contract():
    job = json.loads((ROOT / "colab/jobs/wp05u_u2a_olmo3_7b_coarse_a100.json").read_text())
    assert job["model"] == "allenai/Olmo-3-1025-7B"
    assert job["families"] == ["truthy_answer"]
    assert len(job["revisions"]) == 7
    assert "discovery only" in job["claim_boundary"]


def test_scale_sweep_model_argument_is_wired():
    text = (ROOT / "colab/gsd_remote_job.py").read_text()
    assert 'job.get("model"' in text
    assert '"--model"' in text


def test_stable_controls_model_argument_is_wired():
    text = (ROOT / "colab/gsd_remote_job.py").read_text()
    branch = text.split('if workload == "gsd_wp02_stable_controls":', 1)[1].split('elif workload ==', 1)[0]
    assert '"--model"' in branch
    assert 'job.get("model"' in branch
