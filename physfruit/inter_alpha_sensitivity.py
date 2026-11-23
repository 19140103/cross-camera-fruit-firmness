"""Training-fold-independent sensitivity of the fixed paired-Ridge penalty."""
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


def run(fruit: str, camera_a: str, camera_b: str,
        alphas=(10.0, 20.0, 40.0, 100.0, 200.0, 500.0),
        repeats: int = 5, folds: int = 5,
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows: list[dict] = []
    for alpha in alphas:
        for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
            for direction, src, tgt in [
                (f"{camera_a}->{camera_b}", A, B),
                (f"{camera_b}->{camera_a}", B, A),
            ]:
                pred_all = np.empty(len(y0))
                for tr, te in splits:
                    reg = Pipeline([
                        ("scale", StandardScaler()),
                        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                              max_iter=100000)),
                    ])
                    reg.fit(src[tr], y[tr])
                    mapped = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                    pred_all[te] = np.expm1(reg.predict(mapped))
                rmse = float(np.sqrt(mean_squared_error(y0, pred_all)))
                rows.append(dict(
                    fruit=fruit, direction=direction, alpha=float(alpha),
                    repeat=rep, aggregate="repeat_pooled",
                    R2=float(r2_score(y0, pred_all)), RMSE=rmse,
                    MAE=float(mean_absolute_error(y0, pred_all)),
                    RPD=float(np.std(y0, ddof=1) / rmse if rmse else np.inf),
                    n_test=len(y0), n_common_bands=A.shape[1],
                ))
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_alpha_sensitivity_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, saved={path}")
    print(out.groupby(["direction", "alpha"])[["R2", "RMSE", "MAE", "RPD"]]
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
    run(args.fruit, args.a, args.b, repeats=args.repeats, folds=args.folds,
        cache=Path(args.cache))
