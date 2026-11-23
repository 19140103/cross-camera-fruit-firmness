"""Sensitivity of paired-camera translation to calibration-pair budget.

The source hardness model is fitted with every labelled source-camera training
fruit in an outer split.  Only the paired training fruits used for the Ridge
translation are subsampled.  This separates the cost of collecting paired
calibration spectra from the cost of building the source model.
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
            "MAE": float(mean_absolute_error(y, pred)),
            "RPD": float(np.std(y, ddof=1) / rmse) if rmse else np.inf}


def source_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000)),
    ])


def run(fruit: str, camera_a: str, camera_b: str, repeats: int = 5,
        folds: int = 5, alpha: float = 100.0, n_subsamples: int = 5,
        budgets: tuple[int, ...] = (4, 8, 16, 24, 32),
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows: list[dict] = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt, src_name, tgt_name in [
            (f"{camera_a}->{camera_b}", A, B, camera_a, camera_b),
            (f"{camera_b}->{camera_a}", B, A, camera_b, camera_a),
        ]:
            for budget in budgets:
                pred_reps: list[np.ndarray] = []
                n_cal_max = 0
                for sample_rep in range(n_subsamples):
                    pred_all = np.empty(len(y0))
                    for fold, (tr, te) in enumerate(splits):
                        reg = source_model().fit(src[tr], y[tr])
                        n_cal = min(int(budget), len(tr))
                        n_cal_max = max(n_cal_max, n_cal)
                        if n_cal == len(tr):
                            cal = tr
                        else:
                            rng = np.random.default_rng(
                                140729 + rep * 10000 + fold * 100 + sample_rep)
                            cal = rng.choice(tr, size=n_cal, replace=False)
                        mapped = fit_pair_ridge(src[cal], tgt[cal], tgt[te], alpha)
                        pred_all[te] = np.expm1(reg.predict(mapped))
                    pred_reps.append(pred_all)
                    row = metric(y0, pred_all)
                    row.update(fruit=fruit, direction=direction,
                               method="pair_ridge", budget_requested=int(budget),
                               n_calibration_pairs=int(n_cal_max),
                               sample_rep=sample_rep, repeat=rep, fold=-1,
                               aggregate="repeat_pooled", n_test=len(y0),
                               n_common_bands=A.shape[1], alpha=alpha,
                               source_camera=src_name, target_camera=tgt_name)
                    rows.append(row)
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_calibration_budget_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: paired {camera_a}<->{camera_b}, n={len(p)}")
    print(out.groupby(["direction", "budget_requested"])[["R2", "RMSE", "MAE", "RPD"]]
          .agg(["mean", "std"]).round(4).to_string())
    print(f"saved: {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--subsamples", type=int, default=5)
    ap.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 24, 32])
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.repeats, args.folds, args.alpha,
        args.subsamples, tuple(args.budgets), Path(args.cache))
