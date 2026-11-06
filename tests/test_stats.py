import pytest

from fc_distill.stats import interval, mcnemar, metric_block, wilson


def test_wilson_matches_known_values_and_stays_in_bounds():
    low, high = wilson(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson(50, 100)
    assert (low, high) == (pytest.approx(0.4038, abs=1e-4), pytest.approx(0.5962, abs=1e-4))
    assert wilson(10, 10)[1] == 1.0


def test_interval_uses_wilson_for_binary_and_normal_otherwise():
    mean, low, high = interval([1.0, 0.0, 1.0, 1.0])
    assert (mean, low, high) == (0.75, *wilson(3, 4))
    mean, low, high = interval([0.5, 0.5, 0.5])
    assert (mean, low, high) == (0.5, 0.5, 0.5)


def test_metric_block_handles_empty_rows():
    assert metric_block([], ["exact"]) == {"n": 0, "exact": None, "exact_ci": None}
    block = metric_block([{"exact": True}, {"exact": False}], ["exact"])
    assert block["exact"] == 0.5
    assert block["exact_ci"] == [round(x, 4) for x in wilson(1, 2)]


def test_mcnemar_counts_discordant_pairs_and_exact_p():
    a = [True, True, False, False, False, False, False, True]
    b = [True, False, True, True, True, True, True, True]
    out = mcnemar(a, b)
    assert (out["only_a_right"], out["only_b_right"], out["n"]) == (1, 5, 8)
    # binomial(6, 0.5): P(X <= 1) = 7/64, two-sided 14/64
    assert out["p"] == pytest.approx(14 / 64)


def test_mcnemar_is_one_when_nothing_differs_or_split_evenly():
    assert mcnemar([True, False], [True, False])["p"] == 1.0
    assert mcnemar([True, False], [False, True])["p"] == 1.0


def test_mcnemar_rejects_unpaired_input():
    with pytest.raises(ValueError, match="same length"):
        mcnemar([True], [True, False])
