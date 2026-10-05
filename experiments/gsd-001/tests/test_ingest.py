from pathlib import Path

from gsd.ingest import ingest_checkpoint


def test_ingest_locked_style_output():
    root = Path(__file__).parent / "fixtures" / "upstream"
    rows, summary = ingest_checkpoint(root, "stage1-step20000-tokens42B")
    assert len(rows) == 2
    assert rows[0]["sum_margin"] == 1.0
    assert rows[1]["sum_margin"] == 0.7
    assert summary["step"] == 20000
    assert summary["tasks"]["flipped_answer/sst2"]["state"] == "GENERALIZING"
    assert summary["families"]["flipped_answer"]["state"] == "GENERALIZING"
    assert "Task summaries are primary" in summary["aggregation_boundary"]
