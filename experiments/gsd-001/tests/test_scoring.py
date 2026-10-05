import json
from pathlib import Path

from gsd.scoring import (
    STATE_GENERALIZING,
    classify_margins,
    record_from_pair_scores,
    resolve_answer_start,
    span_score_from_token_logprobs,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        table = {
            "Q: 2+2? A: ": [1, 2, 3],
            "Q: 2+2? A: 4": [1, 2, 3, 4],
            "Q: 2+2? A:": [1, 2],
        }
        return table[text]


def test_stable_answer_boundary():
    ids, start, mode = resolve_answer_start(FakeTokenizer(), "Q: 2+2? A:", "4")
    assert ids == [1, 2, 3, 4]
    assert start == 3
    assert mode == "stable_prompt_space"


class MergeTokenizer:
    def encode(self, text, add_special_tokens=False):
        table = {
            "Q: 2+2? A: ": [1, 2, 3],
            "Q: 2+2? A: 4": [1, 9],
            "Q: 2+2? A:": [1, 2],
            " 4": [8],
        }
        return table[text]


def test_forced_continuation_boundary():
    ids, start, mode = resolve_answer_start(MergeTokenizer(), "Q: 2+2? A:", "4")
    assert ids == [1, 2, 8]
    assert start == 2
    assert mode == "forced_separate_continuation"


def test_golden_score_arithmetic():
    fixture = json.loads(
        (Path(__file__).parents[1] / "golden_fixture.json").read_text()
    )
    margins = []
    for case in fixture["cases"][:2]:
        intelligence = span_score_from_token_logprobs(case["intelligence_logps"])
        parrot = span_score_from_token_logprobs(case["parrot_logps"])
        row = record_from_pair_scores(intelligence, parrot, id=case["id"])
        assert abs(row["sum_margin"] - case["expected_sum_margin"]) < 1e-12
        margins.append(row["sum_margin"])
    assert classify_margins(margins)["state"] == STATE_GENERALIZING
