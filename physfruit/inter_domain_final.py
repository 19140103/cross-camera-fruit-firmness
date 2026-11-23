"""Predeclared final evaluation for paired-camera spectral translation.

The regularisation rule is fixed before reading any test metrics:
    alpha = 5 * number_of_common_bands

For the current 20-band overlap grid this gives alpha=100. The rule is a
simple high-dimensional shrinkage choice (p=20, n_train≈22--32) and avoids
selecting alpha from the outer test results. This script reports only the
methods designated for the final comparison.
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
from physfruit.inter_alignment import align
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table

ROOT = Path(__file__).resolve().parent.parent


def metric(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return {"R2": float(r2_score(y, p)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, p)),
            "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf)}


def run(fruit, camera_a, camera_b, repeats=5, folds=5,
        cache=ROOT / "baseline_cache"):
    A, B, p = paired_table(fruit, camera_a, camera_b)
    alpha = 5.0 * A.shape[1]
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt, source_name, target_name in [
                (f"{camera_a}->{camera_b}", A, B, camera_a, camera_b),
                (f"{camera_b}->{camera_a}", B, A, camera_b, camera_a)]:
            pred_by_method = {}
            for method in ("raw", "scale", "coral", "pair_ridge"):
                pred_all = np.empty(len(y0))
                for fold, (tr, te) in enumerate(splits):
                    reg = Pipeline([
                        ("scale", StandardScaler()),
                        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                              max_iter=100000)),
                    ])
                    reg.fit(src[tr], y[tr])
                    if method == "pair_ridge":
                        target_test = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                    else:
                        target_test = align(src[tr], tgt[tr], tgt[te], method)
                    pred = np.expm1(reg.predict(target_test))
                    pred_all[te] = pred
                    row = metric(y0[te], pred)
                    row.update(fruit=fruit, direction=direction, method=method,
                               repeat=rep, fold=fold, aggregate="fold",
                               n_test=len(te), n_common_bands=A.shape[1],
                               alpha=alpha)
                    rows.append(row)
                row = metric(y0, pred_all)
                row.update(fruit=fruit, direction=direction, method=method,
                           repeat=rep, fold=-1, aggregate="repeat_pooled",
                           n_test=len(y0), n_common_bands=A.shape[1], alpha=alpha)
                rows.append(row)
                pred_by_method[method] = pred_all
            # Paired prediction differences are useful diagnostics, but not
            # treated as independent observations or significance tests.
            for method in ("scale", "coral", "pair_ridge"):
                delta = metric(y0, pred_by_method[method])['R2'] - metric(y0, pred_by_method['raw'])['R2']
                rows.append(dict(fruit=fruit, direction=direction, method=f"delta_{method}_vs_raw",
                                 repeat=rep, fold=-1, aggregate="repeat_delta",
                                 n_test=len(y0), n_common_bands=A.shape[1], alpha=alpha,
                                 R2=delta, RMSE=np.nan, MAE=np.nan, RPD=np.nan))
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_domain_final_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    q = out[out["aggregate"] == "repeat_pooled"]
    print(f"{fruit}: {camera_a}<->{camera_b}, n={len(p)}, alpha=5*p={alpha:.1f}")
    print(q.groupby(["direction", "method"])[["R2", "RMSE", "MAE", "RPD"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("repeat-level R2 changes versus raw:")
    print(out[out["aggregate"] == "repeat_delta"].groupby(["direction", "method"])["R2"]
          .agg(["mean", "std", "min", "max"]).round(4).to_string())
    print(f"saved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.repeats, args.folds, Path(args.cache))
