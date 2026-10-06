from run_wp06_r4_continued import (
    BATCH_ITEMS,
    EVAL_ITEMS,
    EVAL_STEPS,
    GRAD_CLIP,
    LEARNING_RATE,
    MAX_STEPS,
    SEEDS,
    TRAIN_ITEMS,
    WEIGHT_DECAY,
)


def test_wp06_r4_frozen_contract():
    assert TRAIN_ITEMS == list(range(128, 192))
    assert EVAL_ITEMS == list(range(192, 256))
    assert SEEDS == (0, 1, 2)
    assert EVAL_STEPS == (0, 1, 2, 4, 8, 16, 32)
    assert MAX_STEPS == 32
    assert BATCH_ITEMS == 2
    assert LEARNING_RATE == 1e-5
    assert WEIGHT_DECAY == 0.0
    assert GRAD_CLIP == 1.0
