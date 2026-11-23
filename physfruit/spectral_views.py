"""Multi-view spectral-shape controls for continuous firmness.

The experiment asks whether a small, auditable representation change can make
the strong raw-spectrum ElasticNet more robust.  Every transform is computed
independently for each spectrum; target-aware tuning is deliberately left out
of this first screening pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.baseline_regression import load_fruit_table
from physfruit.evaluation import repeated_group_folds, grouped_folds


def views(X: np.ndarray) -> dict[str, np.ndarray]:
    # Raw reflectance; smooth shape; first/second derivatives; and SNV.
    sm = savgol_filter(X, 11, 2, axis=1)
    d1 = savgol_filter(X, 15, 2, deriv=1, axis=1)
    d2 = savgol_filter(X, 15, 2, deriv=2, axis=1)
    snv = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return {
        "raw": X,
        "smooth": sm,
        "snv": snv,
        "d1": d1,
        "d2": d2,
        "raw_snv": np.column_stack([X, snv]),
        "raw_d1": np.column_stack([X, d1]),
        "raw_d1_snv": np.column_stack([X, d1, snv]),
        "raw_smooth_d1": np.column_stack([X, sm, d1]),
    }


def metrics(y, pred):
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"R2": float(r2_score(y, pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, pred)),
            "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf)}


def run(fruit="Avocado", camera="VIS", repeats=5, folds=5, cache=Path("baseline_cache")):
    X, d = load_fruit_table(fruit, camera, cache)
    Xs = views(X); y0 = d.firmness_g_cm2.to_numpy(float); y = np.log1p(y0); groups = d.physical_group.to_numpy()
    rows = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=folds)):
        pooled_pred = {name: np.empty(len(y0)) for name in Xs}
        for fold, (tr, te) in enumerate(splits):
            for name, Z in Xs.items():
                for alpha in ([.003, .01, .03] if name in ("raw_d1", "raw_d1_snv") else [.01]):
                    m = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=30000))])
                    m.fit(Z[tr], y[tr]); p = np.expm1(np.asarray(m.predict(Z[te])).reshape(-1)); rm = np.sqrt(mean_squared_error(y0[te], p))
                    rows.append(dict(view=name, alpha=alpha, R2=r2_score(y0[te], p), RMSE=rm, MAE=mean_absolute_error(y0[te], p), RPD=np.std(y0[te], ddof=1)/rm, repeat=rep, fold=fold))
    out = pd.DataFrame(rows); cache.mkdir(parents=True, exist_ok=True); out.to_csv(cache / f"spectral_views_{fruit}_{camera}.csv", index=False)
    print(out.groupby(["view", "alpha"])[["R2", "RMSE", "MAE", "RPD"]].agg(["mean", "std"]).round(4).to_string())


def nested_run(fruit="Avocado", camera="VIS", repeats=5, folds=5, cache=Path("baseline_cache")):
    """Outer GroupKFold with alpha selected only inside each training fold."""
    X, d = load_fruit_table(fruit, camera, cache); Xs = views(X)
    # Keep the confirmatory set compact; the exploratory table above remains
    # available but is not used to claim a tuned improvement.
    Xs = {k: Xs[k] for k in ("raw", "smooth", "raw_d1", "raw_d1_snv")}
    alphas = (.001, .003, .01, .03, .1)
    y0 = d.firmness_g_cm2.to_numpy(float); y = np.log1p(y0); groups = d.physical_group.to_numpy()
    rows = []

    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=folds)):
        pooled_pred = {name: np.empty(len(y0)) for name in Xs}
        for fold, (tr, te) in enumerate(splits):
            # Inner selection is performed independently per representation.
            for name, Z in Xs.items():
                inner_groups = groups[tr]
                scores = []
                for alpha in alphas:
                    inner_pred = np.empty(len(tr))
                    for itr, ite in grouped_folds(inner_groups, folds=3, seed=204000 + rep * 100 + fold):
                        m = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=alpha, l1_ratio=.5, max_iter=30000))])
                        m.fit(Z[tr][itr], y[tr][itr]); inner_pred[ite] = np.expm1(m.predict(Z[tr][ite]))
                    scores.append(-float(np.sqrt(mean_squared_error(y0[tr], inner_pred))))
                best = alphas[int(np.argmax(scores))]
                m = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=best, l1_ratio=.5, max_iter=30000))])
                m.fit(Z[tr], y[tr]); p = np.expm1(m.predict(Z[te])); rm = np.sqrt(mean_squared_error(y0[te], p))
                pooled_pred[name][te] = p
                row = metrics(y0[te], p); row.update(view=name, alpha=best, inner_RMSE=-max(scores), repeat=rep, fold=fold, aggregate="fold", n_test=len(te)); rows.append(row)
        for name, p in pooled_pred.items():
            row = metrics(y0, p); row.update(view=name, alpha="pooled", inner_RMSE=np.nan, repeat=rep, fold=-1, aggregate="repeat_pooled", n_test=len(y0)); rows.append(row)
    out = pd.DataFrame(rows); cache.mkdir(parents=True, exist_ok=True); out.to_csv(cache / f"spectral_views_nested_{fruit}_{camera}.csv", index=False)
    print("\nNested outer evaluation (alpha chosen within each training fold):")
    print(out.groupby("view")[["R2", "RMSE", "MAE", "RPD"]].agg(["mean", "std"]).round(4).to_string())
    print("selected alpha counts:"); print(pd.crosstab(out.view, out.alpha).to_string())
    print("repeat-pooled:"); print(out[out["aggregate"] == "repeat_pooled"].groupby("view")[["R2", "RMSE", "MAE", "RPD"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fruit", default="Avocado"); ap.add_argument("--camera", default="VIS"); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--cache", default="baseline_cache"); ap.add_argument("--nested", action="store_true"); a = ap.parse_args(); (nested_run if a.nested else run)(a.fruit, a.camera, a.repeats, a.folds, Path(a.cache))
