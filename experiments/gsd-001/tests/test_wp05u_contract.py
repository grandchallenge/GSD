from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_wp05u_protocol_preserves_cross_transition_gate():
    text = (ROOT / "WP05U_UTILITY_EXPANSION_PROTOCOL.md").read_text()
    assert "VALIDATE_FROZEN_LAYER13_Q14H2_SIGNATURE_ON_NEXT_INDEPENDENT_CONFIRMED_TRANSITION" in text
    assert 'no universal "head 2" identity claim' in text
    assert "training data -> internal operator movement -> behavioural state movement" in text


def test_wp05u_u1_job_is_same_transition_causal_only():
    job = json.loads((ROOT / "colab/jobs/wp05u_u1_directional_causal_a100.json").read_text())
    assert job["workload"] == "gsd_wp05u_directional_causal"
    assert job["fresh_holdout_claim"] is False
    assert job["cross_transition_predictive_claim"] is False
    assert job["revisions"] == [
        "stage1-step2000-tokens5B",
        "stage1-step3000-tokens7B",
        "stage1-step4000-tokens9B",
    ]
