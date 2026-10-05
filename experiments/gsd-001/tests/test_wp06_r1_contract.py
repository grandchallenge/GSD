from run_wp06_r1_activation import (
    DISCOVERY_ITEMS,
    HELDOUT_ITEMS,
    PATCH_POINTS,
    SOURCE,
    FLANK,
    TARGET,
)


def test_wp06_r1_frozen_contract_shape():
    assert PATCH_POINTS == (4, 8, 12, 16)
    assert SOURCE == "stage1-step2000-tokens5B"
    assert FLANK == "stage1-step3000-tokens7B"
    assert TARGET == "stage1-step4000-tokens9B"
    assert DISCOVERY_ITEMS == list(range(0, 64))
    assert HELDOUT_ITEMS == list(range(64, 128))
