"""Nested training-only alpha selection for paired-camera translation.

The outer test labels are never used to choose the Ridge regularisation.  For
each outer training fold, alpha is selected by grouped reconstruction error on
the paired spectra, then the selected mapping is evaluated once on the outer
test spectra.
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
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table, select_pair_alpha

ROOT = Path(__file__).resolve().parent.parent


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y, pred)),
        "RPD": float(np.std(y, ddof=1) / rmse) if rmse else np.inf,
    }


def source_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000)),
    ])


def run(fruit: str, camera_a: str, camera_b: str, repeats: int = 5,
        folds: int = 5, candidates=(1.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
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
            pred_all = np.empty(len(y0))
            chosen = []
            for fold, (tr, te) in enumerate(splits):
                alpha, recon = select_pair_alpha(src[tr], tgt[tr], groups[tr], candidates=candidates)
                chosen.append(float(alpha))
                reg = source_model().fit(src[tr], y[tr])
                mapped = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                pred = np.expm1(reg.predict(mapped))
                pred_all[te] = pred
                row = metric(y0[te], pred)
                row.update(
                    fruit=fruit, direction=direction, method="pair_ridge_nested",
                    repeat=rep, fold=fold, aggregate="fold", n_test=len(te),
                    n_train=len(tr), n_common_bands=A.shape[1],
                    alpha=float(alpha), train_recon_rmse=float(recon),
                    source_camera=src_name, target_camera=tgt_name,
                )
                rows.append(row)
            row = metric(y0, pred_all)
            row.update(
                fruit=fruit, direction=direction, method="pair_ridge_nested",
                repeat=rep, fold=-1, aggregate="repeat_pooled", n_test=len(y0),
                n_train=int(len(y0) - len(y0) / folds), n_common_bands=A.shape[1],
                alpha_mean=float(np.mean(chosen)), alpha_values="|".join(str(int(x)) for x in chosen),
                source_camera=src_name, target_camera=tgt_name,
            )
            rows.append(row)

    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_nested_alpha_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    q = out[out["aggregate"] == "repeat_pooled"]
    print(f"{fruit}: {camera_a}<->{camera_b}, nested alpha candidates={list(candidates)}")
    print(q.groupby(["direction", "method"])[["R2", "RMSE", "MAE", "RPD"]]
          .agg(["mean", "std"]).round(4).to_string())
    print("chosen alpha counts:")
    print(out[out["aggregate"] == "fold"].groupby("direction")["alpha"].value_counts().to_string())
    print(f"saved: {path}")
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
    run(args.fruit, args.a, args.b, args.repeats, args.folds, cache=Path(args.cache))
