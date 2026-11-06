from fc_distill.eval_tracking import log_evals
from tests.fakes import FakeTracker

XLAM_BLOCK = {
    "n": 3,
    "exact": 0.5,
    "exact_ci": [0.2, 0.8],
    "slices": {
        "call_pattern": {
            "single": {"too_small": False, "n": 2, "exact": 1.0, "exact_ci": [0.3, 1.0]},
            "multiple": {"too_small": True, "n": 0, "exact": None, "exact_ci": None},
        },
        "distractors": {"1-2": {"too_small": True, "n": 1, "exact": 0.0, "exact_ci": [0.0, 0.8]}},
    },
    "seconds": 1.5,
}
BFCL_BLOCK = {
    "overall": {"n": 2, "bfcl_exact": 0.5, "bfcl_exact_ci": [0.1, 0.9]},
    "slices": {"simple": {"n": 1, "bfcl_exact": 1.0}},
    "categories": {"parallel": {"n": 1, "bfcl_exact": 0.0, "below_min_rows": True}},
    "rows_calling_a_train_function_name": 44,
}


def test_every_eval_lands_under_its_own_flat_keys() -> None:
    tracker = FakeTracker()
    log_evals(tracker, {"test-clean": XLAM_BLOCK, "bfcl": BFCL_BLOCK})
    assert tracker.summarised == {
        "test_clean/n": 3,
        "test_clean/exact": 0.5,
        "test_clean/exact_lo": 0.2,
        "test_clean/exact_hi": 0.8,
        "test_clean/seconds": 1.5,
        "test_clean/slice/call_pattern/single/n": 2,
        "test_clean/slice/call_pattern/single/exact": 1.0,
        "test_clean/slice/call_pattern/multiple/n": 0,
        "test_clean/slice/call_pattern/multiple/exact": None,
        "test_clean/slice/distractors/d1_2/n": 1,
        "test_clean/slice/distractors/d1_2/exact": 0.0,
        "bfcl/n": 2,
        "bfcl/bfcl_exact": 0.5,
        "bfcl/bfcl_exact_lo": 0.1,
        "bfcl/bfcl_exact_hi": 0.9,
        "bfcl/rows_calling_a_train_function_name": 44,
        "bfcl/slice/simple/n": 1,
        "bfcl/slice/simple/bfcl_exact": 1.0,
        "bfcl/category/parallel/n": 1,
        "bfcl/category/parallel/bfcl_exact": 0.0,
    }


def test_every_slice_is_a_row_of_one_table() -> None:
    tracker = FakeTracker()
    log_evals(tracker, {"test-clean": XLAM_BLOCK, "bfcl": BFCL_BLOCK})
    columns, rows = tracker.tables["slices"]
    assert columns == [
        "eval", "dim", "bucket", "too_small", "n", "exact", "exact_lo", "exact_hi",
        "bfcl_exact", "below_min_rows",
    ]  # fmt: skip
    assert rows == [
        ["test_clean", "call_pattern", "single", False, 2, 1.0, 0.3, 1.0, None, None],
        ["test_clean", "call_pattern", "multiple", True, 0, None, None, None, None, None],
        ["test_clean", "distractors", "d1_2", True, 1, 0.0, 0.0, 0.8, None, None],
        ["bfcl", "slice", "simple", None, 1, None, None, None, 1.0, None],
        ["bfcl", "category", "parallel", None, 1, None, None, None, 0.0, True],
    ]


def test_bucket_names_become_valid_w_and_b_keys() -> None:
    tracker = FakeTracker()
    buckets = {b: {"n": 1, "exact": 1.0} for b in ("0", "1-2", "3+")}
    log_evals(tracker, {"test": {"n": 3, "slices": {"distractors": buckets}}})
    assert {k for k in tracker.summarised if k.endswith("/n")} == {
        "test/n",
        "test/slice/distractors/d0/n",
        "test/slice/distractors/d1_2/n",
        "test/slice/distractors/d3plus/n",
    }
