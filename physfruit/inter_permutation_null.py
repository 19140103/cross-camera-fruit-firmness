"""Repeated paired-identity permutation null for cross-camera transfer.

For every physical-fruit outer fold, the target spectra in the *training*
pairs are independently permuted before fitting the Ridge mapper. Test spectra,
source labels and the outer split remain untouched. Each permutation is then
pooled across the five test folds, giving an empirical null distribution for
R2 and Delta R2 relative to raw direct transfer. The permutation replicates are
descriptive null draws, not independent biological replicates or significance
tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.evaluation import repeated_group_folds
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table

ROOT = Path(__file__).resolve().parent.parent


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"R2": float(r2_score(y, pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, pred))}


def run(fruit: str, camera_a: str, camera_b: str, repeats: int = 5,
        folds: int = 5, n_permutations: int = 50,
        cache: Path = ROOT / "baseline_cache", seed: int = 20261004) -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    alpha_pair = 5.0 * A.shape[1]
    rows: list[dict] = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt in [
            (f"{camera_a}->{camera_b}", A, B),
            (f"{camera_b}->{camera_a}", B, A),
        ]:
            pred_raw = np.empty(len(y0))
            pred_true = np.empty(len(y0))
            pred_perm = [np.empty(len(y0)) for _ in range(n_permutations)]
            for fold, (tr, te) in enumerate(splits):
                reg = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                          max_iter=100000)),
                ]).fit(src[tr], y[tr])
                pred_raw[te] = np.expm1(reg.predict(tgt[te]).ravel())
                pred_true[te] = np.expm1(reg.predict(
                    fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha_pair)).ravel())
                # One independent within-fold permutation per null draw.  The
                # seed includes direction, repeat, fold and draw so every draw
                # is reproducible while remaining independent across folds.
                for perm_id in range(n_permutations):
                    rng = np.random.default_rng(
                        seed + 100000 * rep + 10000 * (direction != f"{camera_a}->{camera_b}")
                        + 100 * fold + perm_id)
                    shuffled = rng.permutation(len(tr))
                    mapped = fit_pair_ridge(src[tr], tgt[tr][shuffled],
                                            tgt[te], alpha_pair)
                    pred_perm[perm_id][te] = np.expm1(reg.predict(mapped).ravel())
            raw_m = metric(y0, pred_raw)
            true_m = metric(y0, pred_true)
            for perm_id, pred in enumerate(pred_perm):
                m = metric(y0, pred)
                m.update(fruit=fruit, direction=direction, repeat=rep,
                         perm_id=perm_id, aggregate="repeat_pooled",
                         n_test=len(y0), n_common_bands=A.shape[1],
                         alpha=alpha_pair, R2_raw=raw_m["R2"],
                         R2_true_pair=true_m["R2"],
                         delta_R2=m["R2"] - raw_m["R2"],
                         delta_R2_true=true_m["R2"] - raw_m["R2"])
                rows.append(m)
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_permutation_null_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    q = out.groupby("direction")["delta_R2"].agg(
        mean="mean", std="std", q025=lambda x: x.quantile(.025),
        median="median", q975=lambda x: x.quantile(.975))
    # The observed value is the mean of the five repeat-pooled true-pair
    # deltas.  Null draws remain pooled across repeats for an empirical
    # distribution; the fraction is descriptive and is not a p-value.
    observed = out.groupby(["direction", "repeat"])["delta_R2_true"].first().groupby("direction").mean()
    q["observed_delta_R2_true"] = observed
    q["null_ge_observed_fraction"] = out.groupby("direction").apply(
        lambda g: float(np.mean(g["delta_R2"] >= observed.loc[g.name])),
        include_groups=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, saved={path}")
    print(q.round(4).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=50)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    ap.add_argument("--seed", type=int, default=20261004)
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.repeats, args.folds,
        args.permutations, Path(args.cache), args.seed)
