from run_wp06_r3_parameter import (
    ALPHA,
    BATCH_ITEMS,
    CONTROL_N,
    LEARNING_RATE,
    RANK,
    RESERVE_ITEMS,
    SEEDS,
    TRAIN_ITEMS,
    TRAIN_STEPS,
)


def test_wp06_r3_frozen_contract():
    assert RANK == 1
    assert ALPHA == 1.0
    assert TRAIN_ITEMS == list(range(128, 192))
    assert RESERVE_ITEMS == list(range(256, 287))
    assert SEEDS == (0, 1, 2)
    assert TRAIN_STEPS == 32
    assert BATCH_ITEMS == 2
    assert LEARNING_RATE == 3e-3
    assert CONTROL_N == 32
