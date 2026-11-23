"""Unlabelled transfer-reliability diagnostics for a future applicability gate.

The descriptors are computed from training spectra only: nearest distance and
Mahalanobis-like leverage of translated target spectra relative to the source
calibration cloud, plus the translation magnitude.  Test hardness is used only
afterwards to audit whether these descriptors predict failure; it is never used
to fit or select a gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
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
    groups = p.uid.to_numpy()
    rows: list[dict] = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
        for direction, src, tgt in [
            (f"{camera_a}->{camera_b}", A, B),
            (f"{camera_b}->{camera_a}", B, A),
        ]:
            for fold, (tr, te) in enumerate(splits):
                reg = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                          max_iter=100000)),
                ])
                reg.fit(src[tr], y[tr])
                mapped = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
                pair_pred = np.expm1(reg.predict(mapped))
                mu = src[tr].mean(0)
                sd = src[tr].std(0) + 1e-6
                z_train = (src[tr] - mu) / sd
                z_mapped = (mapped - mu) / sd
                z_target = (tgt[te] - mu) / sd
                nbrs = NearestNeighbors(n_neighbors=1).fit(z_train)
                nn_dist = nbrs.kneighbors(z_mapped, return_distance=True)[0][:, 0]
                cov = np.cov(z_train, rowvar=False) + 1e-3 * np.eye(z_train.shape[1])
                inv_cov = np.linalg.pinv(cov)
                centered = z_mapped - z_train.mean(0)
                mahal = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", centered, inv_cov, centered)))
                translation = np.sqrt(np.mean((z_mapped - z_target) ** 2, axis=1))
                for i, idx in enumerate(te):
                    rows.append(dict(
                        fruit=fruit, direction=direction, repeat=rep,
                        fold=fold, uid=groups[idx],
                        true_firmness=y0[idx], prediction=pair_pred[i],
                        abs_error=abs(pair_pred[i] - y0[idx]),
                        nn_distance=float(nn_dist[i]),
                        mahal_distance=float(mahal[i]),
                        translation_magnitude=float(translation[i]),
                        alpha=alpha, n_train=len(tr), n_test=len(te),
                    ))
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"transfer_reliability_diagnostic_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, saved={path}")
    print(out.groupby("direction")[["abs_error", "nn_distance", "mahal_distance",
                                     "translation_magnitude"]].mean().round(4).to_string())
    print("fruit/descriptor means:")
    print(out.groupby(["direction", "fruit"])[["abs_error", "nn_distance", "mahal_distance",
                                                "translation_magnitude"]].mean().round(4).to_string())
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
