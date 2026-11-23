"""Strong calibration-transfer baselines under the frozen outer protocol.

Adds piecewise direct standardisation (PDS), a classical paired-spectra
calibration-transfer method, without changing the frozen main-table protocol.
The PDS window is declared before reading hardness metrics; widths 1 and 2 are
reported as a small method-sensitivity comparison.
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
from physfruit.inter_domain_invariant import fit_pair_ridge, fit_pds, paired_table

ROOT = Path(__file__).resolve().parent.parent


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"R2": float(r2_score(y, pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, pred)),
            "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf)}


def run(fruit: str, camera_a: str, camera_b: str,
        repeats: int = 5, folds: int = 5,
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    alpha_pair = 5.0 * A.shape[1]
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    methods = ("raw", "scale", "coral", "pair_ridge", "pds_w1", "pds_w2")
    rows: list[dict] = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt in [
            (f"{camera_a}->{camera_b}", A, B),
            (f"{camera_b}->{camera_a}", B, A),
        ]:
            pred_all = {m: np.empty(len(y0)) for m in methods}
            for fold, (tr, te) in enumerate(splits):
                reg = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                          max_iter=100000)),
                ])
                reg.fit(src[tr], y[tr])
                mapped = {
                    "raw": tgt[te],
                    "scale": align(src[tr], tgt[tr], tgt[te], "scale"),
                    "coral": align(src[tr], tgt[tr], tgt[te], "coral"),
                    "pair_ridge": fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha_pair),
                    "pds_w1": fit_pds(src[tr], tgt[tr], tgt[te], window=1),
                    "pds_w2": fit_pds(src[tr], tgt[tr], tgt[te], window=2),
                }
                for method, Xte in mapped.items():
                    pred = np.expm1(reg.predict(Xte))
                    pred_all[method][te] = pred
                    row = metric(y0[te], pred)
                    row.update(fruit=fruit, direction=direction, method=method,
                               repeat=rep, fold=fold, aggregate="fold",
                               n_test=len(te), n_common_bands=A.shape[1],
                               pair_alpha=alpha_pair,
                               pds_window=(int(method[-1]) if method.startswith("pds") else np.nan))
                    rows.append(row)
            for method, pred in pred_all.items():
                row = metric(y0, pred)
                row.update(fruit=fruit, direction=direction, method=method,
                           repeat=rep, fold=-1, aggregate="repeat_pooled",
                           n_test=len(y0), n_common_bands=A.shape[1],
                           pair_alpha=alpha_pair,
                           pds_window=(int(method[-1]) if method.startswith("pds") else np.nan))
                rows.append(row)
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_transfer_baselines_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    q = out[out["aggregate"] == "repeat_pooled"]
    print(f"{fruit}: {camera_a}<->{camera_b}, saved={path}")
    print(q.groupby(["direction", "method"])[["R2", "RMSE", "MAE", "RPD"]]
          .agg(["mean", "std"]).round(4).to_string())
    return out


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
