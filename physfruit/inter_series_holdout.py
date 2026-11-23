"""Leave-one-group-out evaluation for paired-camera transfer.

This is a stress test for whether paired spectral translation transfers beyond
the storage series used to estimate the camera mapping.  Each test series is
held out as a whole; source hardness labels and paired target spectra from the
remaining series are used for fitting, while test hardness is used only for
the final metrics.

The default grouping variable is storage ``series``.  The public paired-camera
overlap currently contains only one series per fruit, so a day-level holdout
can be requested as a secondary temporal stress test.  A group with only one
fruit is retained but its metrics should be read as high-variance diagnostics.
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

from physfruit.inter_alignment import align
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table

ROOT = Path(__file__).resolve().parent.parent


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {
        "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y, pred)),
        "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf),
    }


def run(fruit: str, camera_a: str, camera_b: str,
        group_col: str = "series",
        ridge_alpha: float | None = None,
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    if group_col not in p.columns:
        raise ValueError(f"paired table must include {group_col!r}")
    groups = p[group_col].astype(str).to_numpy()
    unique_groups = np.array(sorted(pd.unique(groups)))
    if len(unique_groups) < 2:
        raise ValueError(
            f"{fruit} has only one {group_col} ({unique_groups[0]}); "
            "group holdout is not identifiable")

    alpha = float(ridge_alpha if ridge_alpha is not None else 5.0 * A.shape[1])
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    rows: list[dict] = []

    pooled_pred: dict[tuple[str, str], np.ndarray] = {
        (direction, method): np.full(len(y0), np.nan)
        for direction in (f"{camera_a}->{camera_b}", f"{camera_b}->{camera_a}")
        for method in ("raw", "scale", "coral", "pair_ridge")
    }
    for held_out in unique_groups:
        te = np.flatnonzero(groups == held_out)
        tr = np.flatnonzero(groups != held_out)
        if len(tr) < 2:
            raise ValueError(f"invalid {group_col} split for {fruit}: {held_out}")
        for direction, src, tgt, src_name, tgt_name in [
            (f"{camera_a}->{camera_b}", A, B, camera_a, camera_b),
            (f"{camera_b}->{camera_a}", B, A, camera_b, camera_a),
        ]:
            reg = Pipeline([
                ("scale", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                      max_iter=100000)),
            ])
            reg.fit(src[tr], y[tr])
            for method in ("raw", "scale", "coral", "pair_ridge"):
                if method == "pair_ridge":
                    target_test = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                else:
                    target_test = align(src[tr], tgt[tr], tgt[te], method)
                pred = np.expm1(reg.predict(target_test))
                pooled_pred[(direction, method)][te] = pred
                row = metric(y0[te], pred)
                row.update(
                    fruit=fruit, direction=direction, method=method,
                    aggregate="group",
                    group_col=group_col, held_out_group=held_out,
                    n_train=len(tr), n_test=len(te), n_groups=len(unique_groups),
                    n_common_bands=A.shape[1],
                    alpha=alpha, source_camera=src_name, target_camera=tgt_name,
                )
                rows.append(row)

    for (direction, method), pred in pooled_pred.items():
        if np.isnan(pred).any():
            raise ValueError(f"missing pooled predictions for {direction}/{method}")
        row = metric(y0, pred)
        row.update(
            fruit=fruit, direction=direction, method=method,
            aggregate="group_pooled", group_col=group_col,
            held_out_group="ALL", n_train=np.nan, n_test=len(y0),
            n_groups=len(unique_groups), n_common_bands=A.shape[1],
            alpha=alpha, source_camera=direction.split("->")[0],
            target_camera=direction.split("->")[1],
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_{group_col}_holdout_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, {group_col}={list(unique_groups)}, "
          f"alpha={alpha:.1f}, saved={path}")
    print(out[out["aggregate"] == "group_pooled"].groupby(["direction", "method"])[["R2", "RMSE", "MAE", "RPD"]]
          .agg(["mean", "std"]).round(4).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--group-col", choices=("series", "day"), default="series")
    ap.add_argument("--ridge-alpha", type=float, default=None)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.group_col, args.ridge_alpha, Path(args.cache))
