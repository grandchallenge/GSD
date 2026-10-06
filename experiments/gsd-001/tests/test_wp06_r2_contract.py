from run_wp06_r2_sparse import (
    DISCOVERY_ITEMS,
    HELDOUT_ITEMS,
    RESERVE_ITEMS,
)
from gsd.sparse_recovery import CANDIDATES


def test_wp06_r2_fresh_split_is_frozen():
    assert DISCOVERY_ITEMS == list(range(128, 192))
    assert HELDOUT_ITEMS == list(range(192, 256))
    assert RESERVE_ITEMS == list(range(256, 287))


def test_wp06_r2_candidate_order_is_frozen():
    assert CANDIDATES == (
        (12, "attention"),
        (12, "mlp"),
        (13, "attention"),
        (13, "mlp"),
        (14, "attention"),
        (14, "mlp"),
        (15, "attention"),
        (15, "mlp"),
        (16, "attention"),
        (16, "mlp"),
    )
