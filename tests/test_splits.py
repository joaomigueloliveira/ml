from fc_distill.splits import assign_groups, build_splits
from tests.factories import make_example


def _examples(n):
    return [make_example(i, query=f"query {i}") for i in range(n)]


def test_splits_have_exact_sizes_and_cover_every_row_once():
    splits = assign_groups(_examples(100), validation_size=10, test_size=20, seed=0)
    assert (len(splits["validation"]), len(splits["test"]), len(splits["train"])) == (10, 20, 70)
    all_ids = splits["train"] + splits["validation"] + splits["test"]
    assert sorted(all_ids) == list(range(100))


def test_splits_are_deterministic_per_seed():
    exs = _examples(100)
    assert assign_groups(exs, 10, 20, seed=0) == assign_groups(exs, 10, 20, seed=0)
    assert assign_groups(exs, 10, 20, seed=0) != assign_groups(exs, 10, 20, seed=1)


def test_same_query_never_spans_two_splits():
    exs = [make_example(i, query="Same Query" if i % 2 else f"q{i}") for i in range(60)]
    splits = assign_groups(exs, validation_size=10, test_size=10, seed=0)
    holders = [name for name, ids in splits.items() if any(i % 2 for i in ids)]
    assert len(holders) == 1


def _with_duplicate_and_code_row():
    exs = _examples(40)
    exs.append(make_example(100, query="query 0"))  # exact duplicate of row 0
    exs.append(
        make_example(
            200,
            query="code",
            params={"x": {"type": "float"}},
            answers=[{"name": "f", "arguments": {"x": "math.pi"}}],
        )
    )
    return exs


def test_train_filters_remove_duplicates_and_code_arguments():
    splits, report = build_splits(_with_duplicate_and_code_row(), 0, 0, seed=0)
    assert 100 not in splits["train"]
    assert 200 not in splits["train"]
    assert report["exact_duplicates_removed"] == 1
    assert report["train_code_arguments_removed"] == 1
    assert report["sizes"] == {"train": 40, "validation": 0, "test": 0}


def test_test_split_keeps_code_arguments_but_not_duplicates():
    splits, report = build_splits(_with_duplicate_and_code_row(), 0, test_size=41, seed=0)
    assert 200 in splits["test"]
    assert 100 not in splits["test"]
    assert report["train_code_arguments_removed"] == 0
