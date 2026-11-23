"""Mechanism diagnostics for paired-camera spectral translation.

For each repeated fruit-level split, compare the source/target spectral
discrepancy before and after paired Ridge translation, then report the pooled
hardness-prediction change for the same outer test predictions.  This is a
diagnostic, not an additional model-selection step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.evaluation import repeated_group_folds
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table

ROOT = Path(__file__).resolve().parent.parent


def run(fruit: str, camera_a: str, camera_b: str,
        repeats: int = 5, folds: int = 5,
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    alpha = 5.0 * A.shape[1]
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows: list[dict] = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt in [
            (f"{camera_a}->{camera_b}", A, B),
            (f"{camera_b}->{camera_a}", B, A),
        ]:
            raw_pred = np.empty(len(y0))
            pair_pred = np.empty(len(y0))
            raw_spec = []
            pair_spec = []
            n_test = []
            for tr, te in splits:
                reg = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                          max_iter=100000)),
                ])
                reg.fit(src[tr], y[tr])
                mapped = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                raw_pred[te] = np.expm1(reg.predict(tgt[te]))
                pair_pred[te] = np.expm1(reg.predict(mapped))

                # Compare spectra in source-training standardised coordinates;
                # this avoids a high-response band dominating the summary.
                mu = src[tr].mean(0)
                sd = src[tr].std(0) + 1e-6
                src_te = (src[te] - mu) / sd
                raw_te = (tgt[te] - mu) / sd
                pair_te = (mapped - mu) / sd
                raw_spec.extend(np.sqrt(np.mean((raw_te - src_te) ** 2, axis=1)))
                pair_spec.extend(np.sqrt(np.mean((pair_te - src_te) ** 2, axis=1)))
                n_test.extend([len(te)])
            raw_r2 = float(r2_score(y0, raw_pred))
            pair_r2 = float(r2_score(y0, pair_pred))
            raw_s = float(np.mean(raw_spec))
            pair_s = float(np.mean(pair_spec))
            rows.append(dict(
                fruit=fruit, direction=direction, repeat=rep,
                aggregate="repeat_pooled", n_test=len(y0), alpha=alpha,
                raw_R2=raw_r2, pair_R2=pair_r2,
                delta_R2=pair_r2 - raw_r2,
                raw_spectral_RMSE=raw_s,
                pair_spectral_RMSE=pair_s,
                spectral_reduction=raw_s - pair_s,
                spectral_reduction_pct=(raw_s - pair_s) / raw_s if raw_s else np.nan,
            ))
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_mechanism_diagnostics_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, saved={path}")
    print(out.groupby("direction")[["raw_R2", "pair_R2", "delta_R2",
                                     "raw_spectral_RMSE", "pair_spectral_RMSE",
                                     "spectral_reduction_pct"]]
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
