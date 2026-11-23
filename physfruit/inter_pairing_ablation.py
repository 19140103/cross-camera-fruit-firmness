"""Negative-control ablation for the paired spectral mapper.

The target spectra in each outer training fold are randomly permuted before
fitting the Ridge map. This preserves the target marginal distribution but
destroys same-fruit pairing. Test spectra and all firmness labels remain
untouched. The control asks whether the gain comes from paired identity rather
than from an arbitrary linear calibration.
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
        folds: int = 5, alpha: float | None = None,
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    alpha = float(5 * A.shape[1] if alpha is None else alpha)
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows: list[dict] = []

    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt in [
            (f"{camera_a}->{camera_b}", A, B),
            (f"{camera_b}->{camera_a}", B, A),
        ]:
            pred_all = {"raw": np.empty(len(y0)),
                        "pair_ridge": np.empty(len(y0)),
                        "unpaired_ridge": np.empty(len(y0))}
            for fold, (tr, te) in enumerate(splits):
                reg = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                         max_iter=100000)),
                ]).fit(src[tr], y[tr])
                rng = np.random.default_rng(20261002 + 1000 * rep + fold)
                shuffled = rng.permutation(len(tr))
                mapped_true = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                mapped_unpaired = fit_pair_ridge(src[tr], tgt[tr][shuffled],
                                                 tgt[te], alpha)
                for method, Xte in {
                    "raw": tgt[te],
                    "pair_ridge": mapped_true,
                    "unpaired_ridge": mapped_unpaired,
                }.items():
                    pred = np.expm1(reg.predict(Xte))
                    pred_all[method][te] = pred
                    row = metric(y0[te], pred)
                    row.update(fruit=fruit, direction=direction, method=method,
                               repeat=rep, fold=fold, aggregate="fold",
                               n_test=len(te), n_common_bands=A.shape[1],
                               alpha=alpha)
                    rows.append(row)
            for method, pred in pred_all.items():
                row = metric(y0, pred)
                row.update(fruit=fruit, direction=direction, method=method,
                           repeat=rep, fold=-1, aggregate="repeat_pooled",
                           n_test=len(y0), n_common_bands=A.shape[1],
                           alpha=alpha)
                rows.append(row)

    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_pairing_ablation_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    q = out[out["aggregate"] == "repeat_pooled"]
    print(q.groupby(["direction", "method"])[["R2", "RMSE", "MAE"]]
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
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.repeats, args.folds,
        args.alpha, Path(args.cache))
