from gsd.control_validation import bootstrap_delta_ci, paired_control_disposition


def _pair(margins, hard, lm):
    return {
        "sentiment": {"margins": margins, "hard_accuracy": hard},
        "arithmetic": {"margins": margins, "hard_accuracy": hard},
        "language_model": {"mean_token_logprobs": lm},
    }


def test_paired_control_passes_when_controls_improve():
    left = _pair([0.0, 0.1, -0.1, 0.0], 0.5, [-3.0, -2.9, -3.1, -3.0])
    right = _pair([0.2, 0.3, 0.1, 0.2], 0.75, [-2.7, -2.8, -2.6, -2.7])
    out = paired_control_disposition(left, right, n_boot=200, seed=0)
    assert out["control_ok"] is True
    assert out["catastrophic_degradation"] is False


def test_paired_control_fails_on_catastrophic_hard_accuracy_drop():
    left = _pair([1.0, 1.1, 0.9, 1.0], 0.9, [-2.0, -2.1, -2.0, -2.1])
    right = _pair([0.8, 0.9, 0.7, 0.8], 0.7, [-1.9, -2.0, -1.9, -2.0])
    out = paired_control_disposition(left, right, n_boot=200, seed=0)
    assert out["control_ok"] is False
    assert out["catastrophic_degradation"] is True


def test_bootstrap_delta_is_paired():
    delta, lo, hi = bootstrap_delta_ci(
        [0.0, 1.0, 2.0], [1.0, 2.0, 3.0], n_boot=200, seed=1
    )
    assert delta == 1.0
    assert lo == 1.0
    assert hi == 1.0
