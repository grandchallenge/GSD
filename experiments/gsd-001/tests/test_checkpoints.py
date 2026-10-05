from gsd.checkpoints import (
    first_tranche_20,
    normalize_refs,
    parse_checkpoint_revision,
    stride_refs,
)


def test_parse_revision():
    ref = parse_checkpoint_revision("stage1-step20000-tokens42B")
    assert ref is not None
    assert ref.step == 20000
    assert ref.tokens_b == 42.0


def test_normalize_and_sort_refs():
    refs = normalize_refs([
        "main",
        "stage1-step20000-tokens42B",
        "stage1-step0-tokens0B",
        "stage1-step10000-tokens21B",
    ])
    assert [r.step for r in refs] == [0, 10000, 20000]


def test_stride_refs_includes_final_checkpoint():
    refs = normalize_refs([
        f"stage1-step{i * 1000}-tokens{i}B" for i in range(37)
    ])
    selected = stride_refs(refs, 2)
    assert len(selected) == 19
    assert selected[0].step == 0
    assert selected[-1].step == 36000


def test_first_tranche_20_matches_live_series_shape():
    names = []
    for i in range(37):
        step = i * 1000
        names.append(f"stage1-step{step}-tokens{i}B")
    refs = normalize_refs(names)
    selected = first_tranche_20(refs)
    assert len(selected) == 20
    assert [r.step for r in selected] == [
        0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000,
        20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 35000, 36000,
    ]
