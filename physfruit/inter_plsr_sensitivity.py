"""PLSR source-model sensitivity for paired-camera transfer.

The frozen outer protocol is reused verbatim: physical-fruit grouped repeated
5-fold evaluation, 20 common bands, source labels only for fitting the source
model, and a fixed paired-Ridge translation with alpha=5*p.  The experiment
changes only the source-camera predictor from Elastic Net to PLS regression.
All component counts are reported as a sensitivity analysis rather than tuned
on the outer test folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
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


def source_model(model_name: str, n_components: int | None = None):
    if model_name == "elastic_net":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                  max_iter=100000)),
        ])
    if model_name == "plsr":
        if n_components is None:
            raise ValueError("PLSR requires n_components")
        # StandardScaler is fitted inside each outer training fold.  PLSR is
        # then run without a second scale operation.
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=int(n_components),
                                     scale=False, max_iter=1000)),
        ])
    raise ValueError(model_name)


def run(fruit: str, camera_a: str, camera_b: str, repeats: int = 5,
        folds: int = 5, components=(1, 2, 5, 10, 15),
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
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
            for method, k in [("elastic_net", np.nan)] + [
                ("plsr", int(c)) for c in components
            ]:
                pred_raw = np.empty(len(y0))
                pred_pair = np.empty(len(y0))
                for fold, (tr, te) in enumerate(splits):
                    reg = source_model(method, None if method == "elastic_net" else int(k))
                    reg.fit(src[tr], y[tr])
                    Xraw = tgt[te]
                    Xpair = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha_pair)
                    pred_raw[te] = np.expm1(reg.predict(Xraw).ravel())
                    pred_pair[te] = np.expm1(reg.predict(Xpair).ravel())
                for transfer, pred in [("raw", pred_raw), ("pair_ridge", pred_pair)]:
                    row = metric(y0, pred)
                    row.update(fruit=fruit, direction=direction, source_model=method,
                               n_components=k, transfer=transfer, repeat=rep,
                               aggregate="repeat_pooled", n_test=len(y0),
                               n_common_bands=A.shape[1], pair_alpha=alpha_pair)
                    rows.append(row)
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_plsr_sensitivity_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, saved={path}")
    print(out.groupby(["direction", "source_model", "n_components", "transfer"])
          [["R2", "RMSE", "MAE"]].agg(["mean", "std"]).round(4).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--components", type=int, nargs="+", default=[1, 2, 5, 10, 15])
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.repeats, args.folds,
        tuple(args.components), Path(args.cache))
