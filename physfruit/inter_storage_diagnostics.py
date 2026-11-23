"""Per-storage-day diagnostics for the forward postharvest holdout.

The split and models match ``inter_storage_holdout.py``.  This companion
output records hardness distribution shift and day-wise prediction bias/MAE,
so a negative pooled R2 can be interpreted together with the actual storage
trajectory rather than as a single undifferentiated score.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.inter_alignment import align
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table

ROOT = Path(__file__).resolve().parent.parent


def run(fruit: str, camera_a: str, camera_b: str,
        train_fraction: float = 0.6, alpha: float = 100.0,
        cache: Path = ROOT / "baseline_cache") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    days = np.sort(p.storage_days.astype(float).unique())
    n_train_days = max(2, int(np.ceil(len(days) * train_fraction)))
    n_train_days = min(n_train_days, len(days) - 1)
    cutoff = float(days[n_train_days - 1])
    tr = np.flatnonzero(p.storage_days.to_numpy(float) <= cutoff)
    te = np.flatnonzero(p.storage_days.to_numpy(float) > cutoff)
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    day_values = p.storage_days.to_numpy(float)
    pred_rows: list[dict] = []
    dist_rows: list[dict] = []
    # Distribution report is direction-independent and includes every paired
    # fruit in the chronological split.
    for split_name, idx in [("early_train", tr), ("late_test", te)]:
        for day in days:
            ii = idx[day_values[idx] == day]
            if len(ii) == 0:
                continue
            vals = y0[ii]
            dist_rows.append(dict(fruit=fruit, split=split_name, storage_day=float(day),
                                  n=len(ii), firmness_mean=float(np.mean(vals)),
                                  firmness_sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                                  firmness_min=float(np.min(vals)), firmness_max=float(np.max(vals)),
                                  cutoff_storage_day=cutoff))
    for direction, src, tgt, src_name, tgt_name in [
        (f"{camera_a}->{camera_b}", A, B, camera_a, camera_b),
        (f"{camera_b}->{camera_a}", B, A, camera_b, camera_a),
    ]:
        reg = Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000)),
        ]).fit(src[tr], y[tr])
        mapped = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
        day_model = Ridge(alpha=1.0).fit(day_values[tr, None], y[tr])
        preds = {
            "raw_transfer": np.expm1(reg.predict(tgt[te])),
            "pair_ridge_transfer": np.expm1(reg.predict(mapped)),
            "storage_day_ridge": np.expm1(day_model.predict(day_values[te, None])),
        }
        for method, pred in preds.items():
            for j, ix in enumerate(te):
                pred_rows.append(dict(fruit=fruit, direction=direction, method=method,
                                      storage_day=float(day_values[ix]), uid=p.iloc[ix].uid,
                                      true_firmness=float(y0[ix]), prediction=float(pred[j]),
                                      error=float(pred[j] - y0[ix]), abs_error=float(abs(pred[j] - y0[ix])),
                                      cutoff_storage_day=cutoff, n_train=len(tr), n_test=len(te),
                                      n_storage_days=len(days), n_common_bands=A.shape[1], alpha=alpha,
                                      source_camera=src_name, target_camera=tgt_name))
    preds_df = pd.DataFrame(pred_rows)
    summary = (preds_df.groupby(["fruit", "direction", "method", "storage_day"], as_index=False)
               .agg(n=("true_firmness", "size"), true_mean=("true_firmness", "mean"),
                    true_sd=("true_firmness", "std"), prediction_mean=("prediction", "mean"),
                    prediction_sd=("prediction", "std"), MAE=("abs_error", "mean"),
                    bias=("error", "mean")))
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    base = cache / f"inter_storage_diagnostics_{fruit}_{camera_a}_{camera_b}"
    preds_df.to_csv(str(base) + "_predictions.csv", index=False)
    summary.to_csv(str(base) + "_summary.csv", index=False)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(str(base) + "_distribution.csv", index=False)
    print(f"{fruit}: saved {base}_predictions.csv, _summary.csv, _distribution.csv")
    print("distribution:")
    print(dist.to_string(index=False))
    print("day-wise summary:")
    print(summary.round(3).to_string(index=False))
    return preds_df, summary, dist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.train_fraction, args.alpha, Path(args.cache))
