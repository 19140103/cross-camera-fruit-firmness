"""Critical checks for the fruit-level evaluation contract."""
import numpy as np
import pytest

from physfruit.evaluation import grouped_folds, partition_signature, repeated_group_folds


def test_repeats_actually_change_partitions():
    groups = np.array([f"fruit{i:02d}" for i in range(84)])
    ps = repeated_group_folds(groups, repeats=5)
    assert len({partition_signature(groups, p) for p in ps}) == 5


def test_repeated_days_and_sides_stay_together():
    groups = np.repeat([f"fruit{i:02d}" for i in range(40)], np.arange(40) % 3 + 1)
    for splits in repeated_group_folds(groups, repeats=5):
        assert sorted(np.concatenate([te for _, te in splits])) == list(range(len(groups)))
        for tr, te in splits:
            assert not (set(groups[tr]) & set(groups[te]))


def test_partition_is_reproducible_and_invariant_to_row_order():
    groups = np.repeat([f"fruit{i:02d}" for i in range(30)], 2)
    shuffled = np.random.default_rng(42).permutation(groups)
    assert partition_signature(groups, grouped_folds(groups)) == partition_signature(shuffled, grouped_folds(shuffled))


def test_nested_validation_uses_only_outer_training_groups():
    groups = np.repeat([f"fruit{i:02d}" for i in range(30)], 2)
    for tr, te in grouped_folds(groups):
        for itr, ite in grouped_folds(groups[tr], folds=3):
            assert not (set(groups[tr][itr]) & set(groups[tr][ite]))
            assert not (set(groups[tr][ite]) & set(groups[te]))


def test_insufficient_group_count_is_rejected():
    with pytest.raises(ValueError):
        grouped_folds(["same-fruit"] * 10)
