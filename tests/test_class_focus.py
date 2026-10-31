import numpy as np

from losses import compute_class_weights


def test_focus_indices_raise_weight_for_mild_moderate_and_proliferative():
    counts = np.array([1263, 259, 699, 135, 206], dtype=np.float64)
    weights = compute_class_weights(counts, mode="inverse", focus_indices=[1, 2, 4], focus_multiplier=2.5)

    assert weights.shape == (5,)
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]
    assert weights[4] > weights[0]
    assert weights[1] > 1.0
    assert weights[2] > 1.0
    assert weights[4] > 1.0
