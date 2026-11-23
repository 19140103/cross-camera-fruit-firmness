"""Shared fruit-group evaluation protocol (v2).

Shuffling rows before deterministic GroupKFold does NOT randomise partitions.
Use GroupKFold's explicit group shuffling, verify memberships, and save the
fold assignments with results. Repeats reuse fruits and are not independent
samples for a t-test or naive bootstrap over folds.
"""
from __future__ import annotations

import hashlib
import numpy as np
from sklearn.model_selection import GroupKFold

PROTOCOL = "fruit-group-shuffle-v2"


def grouped_folds(groups, folds=5, seed=20261002):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("Expected a nonempty vector of fruit identities")
    if any(g is None or str(g) in ("", "nan") for g in groups):
        raise ValueError("Missing physical fruit identity")
    k = min(folds, len(np.unique(groups)))
    if k < 2:
        raise ValueError("At least two physical fruit groups are required")
    splitter = GroupKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(groups)), groups=groups))
    for tr, te in splits:
        if set(groups[tr]) & set(groups[te]):
            raise AssertionError("A physical fruit appears in train and test")
    coverage = np.concatenate([te for _, te in splits])
    if not np.array_equal(np.sort(coverage), np.arange(len(groups))):
        raise AssertionError("Each record must be tested exactly once")
    return splits


def partition_signature(groups, splits):
    # Canonical signature independent of row order and fold numbering.
    groups = np.asarray(groups)
    membership = sorted(tuple(sorted(set(groups[te].tolist()))) for _, te in splits)
    return hashlib.sha256(repr(membership).encode("utf-8")).hexdigest()


def repeated_group_folds(groups, repeats=5, folds=5, seed=20261002):
    partitions = [grouped_folds(groups, folds, seed + r) for r in range(repeats)]
    signatures = [partition_signature(groups, p) for p in partitions]
    if len(set(signatures)) != repeats:
        raise ValueError("Repeated CV contains duplicate group partitions")
    return partitions
