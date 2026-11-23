"""Pooled-repeat evaluation for the current representation screening.

R2 is computed after concatenating predictions from all outer test folds in a
repeat. This avoids the severe dependence of fold-wise R2 on the number of
extreme-hardness fruit in one small test fold. Fold-level metrics are retained
in the CSV for audit.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.baseline_regression import load_fruit_table
from physfruit.evaluation import repeated_group_folds, grouped_folds
from physfruit.spectral_views import views


def m(y, p):
    rm = float(np.sqrt(mean_squared_error(y, p)))
    return {"R2": float(r2_score(y, p)), "RMSE": rm,
            "MAE": float(mean_absolute_error(y, p)),
            "RPD": float(np.std(y, ddof=1) / rm)}


def run(repeats=5, folds=5, cache=Path("baseline_cache")):
    X, d = load_fruit_table("Avocado", "VIS", cache); V = views(X)
    V = {k: V[k] for k in ("raw", "smooth", "raw_d1", "raw_d1_snv")}
    # Fixed alpha confirms the representation effect without selecting alpha
    # on the test results; nested tuning is reported separately by spectral_views.
    alphas = {"raw": .01, "smooth": .01, "raw_d1": .03, "raw_d1_snv": .03}
    y0 = d.firmness_g_cm2.to_numpy(float); y = np.log1p(y0); groups = d.physical_group.to_numpy()
    rows = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        pred = {name: np.empty(len(y0)) for name in V}
        for fold, (tr, te) in enumerate(splits):
            for name, Z in V.items():
                model = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=alphas[name], l1_ratio=.5, max_iter=100000))])
                model.fit(Z[tr], y[tr]); p = np.expm1(model.predict(Z[te])); pred[name][te] = p
                row = m(y0[te], p); row.update(view=name, alpha=alphas[name], repeat=rep, fold=fold, aggregate="fold", n_test=len(te)); rows.append(row)
        for name, p in pred.items():
            row = m(y0, p); row.update(view=name, alpha=alphas[name], repeat=rep, fold=-1, aggregate="repeat_pooled", n_test=len(y0)); rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(cache / "corrected_spectral_views.csv", index=False)
    print("Fold-level:"); print(out[out["aggregate"] == "fold"].groupby("view")[['R2','RMSE','MAE','RPD']].agg(['mean','std']).round(4).to_string())
    print("\nRepeat-pooled (one score per repeated group partition):"); print(out[out["aggregate"] == "repeat_pooled"].groupby("view")[['R2','RMSE','MAE','RPD']].agg(['mean','std']).round(4).to_string())


if __name__ == "__main__":
    run()
