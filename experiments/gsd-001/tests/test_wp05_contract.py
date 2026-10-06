from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wp05_protocol_freezes_discovery_and_evaluation():
    text = (ROOT / "WP05_PROTOCOL.md").read_text()
    assert "Rows 0-63 are the WP05 discovery partition" in text
    assert "Rows 64-127 are frozen before execution" in text
    assert "NO_CROSS_TRANSITION_PREDICTIVE_CLAIM" in text


def test_wp05_job_is_exact_three_checkpoint_localization():
    import json
    job = json.loads((ROOT / "colab/jobs/wp05_mechanistic_operator_a100.json").read_text())
    assert job["workload"] == "gsd_wp05_mechanistic_operator_localization"
    assert job["revisions"] == [
        "stage1-step2000-tokens5B",
        "stage1-step3000-tokens7B",
        "stage1-step4000-tokens9B",
    ]
    assert job["fresh_holdout_claim"] is False
    assert job["cross_transition_predictive_claim"] is False
