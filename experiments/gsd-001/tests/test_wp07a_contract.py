from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_wp07a_protocol_preserves_attribution_claim_boundary():
    text = (ROOT / "WP07A_SOURCE_CLASS_ATTRIBUTION_PROTOCOL.md").read_text()
    assert "no exact original-window reconstruction claim" in text
    assert "WP07A_CLASS_ATTRIBUTION_CANDIDATE" in text
    assert "separately frozen WP07B held-out micro-continuation" in text


def test_wp07a_job_is_frozen_source_class_pilot():
    job = json.loads((ROOT / "colab/jobs/wp07a_source_class_attribution_a100.json").read_text())
    assert job["workload"] == "gsd_wp07a_source_class_attribution"
    assert job["resource"]["accelerator"] == "A100"
    assert job["revisions"] == ["stage1-step2000-tokens5B"]
    assert job["training_tokens_per_class"] == 262144
    assert job["fresh_holdout_claim"] is False
    assert job["source_classes"] == [
        "algebraic-stack",
        "arxiv",
        "dclm",
        "open-web-math",
        "pes2o",
        "starcoder",
        "wiki",
    ]


def test_wp07a_runner_wired():
    text = (ROOT / "colab/gsd_remote_job.py").read_text()
    assert 'workload == "gsd_wp07a_source_class_attribution"' in text
    assert "run_wp07a_source_class_attribution.py" in text


def test_wp07a_script_freezes_training_budget_and_markers():
    text = (ROOT / "run_wp07a_source_class_attribution.py").read_text()
    assert "SEQ_LEN = 4096" in text
    assert "GRAD_ACCUM = 4" in text
    assert "OPT_STEPS = 8" in text
    assert "layer_hidden = 13" in text
    assert "layer_q = 14" in text
    assert "head = 2" in text
