"""Leakage-controlled paired-camera domain adaptation.

VIS is the labelled source and VIS_COR (or another paired camera) is the
unlabelled target.  Each outer test fold estimates every alignment operation
from the paired *training* fruits only.  Target hardness labels are never used
to choose an alignment or fit the source regressor.

Methods
-------
raw      : apply the source model directly to target spectra
scale    : per-band target-to-source mean/standard-deviation matching
coral     : covariance alignment estimated from unlabelled train spectra
pair_ridge: learn a regularised target->source spectral translation from
             paired train spectra, using reconstruction error only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.evaluation import repeated_group_folds
from physfruit.inter_alignment import align
from physfruit.labels import camera_pairs, load_records

ROOT = Path(__file__).resolve().parent.parent


def fit_pair_ridge(train_src: np.ndarray, train_tgt: np.ndarray,
                   test_tgt: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    """Map target spectra into source-camera coordinates using pairs only."""
    # Standardise the input/output domains using training pairs only.  This
    # prevents a high-response band from dominating the ridge translation.
    mu_s, sd_s = train_src.mean(0), train_src.std(0) + 1e-6
    mu_t, sd_t = train_tgt.mean(0), train_tgt.std(0) + 1e-6
    zt = (train_tgt - mu_t) / sd_t
    zs = (train_src - mu_s) / sd_s
    mapper = Ridge(alpha=alpha, fit_intercept=True)
    mapper.fit(zt, zs)
    return mapper.predict((test_tgt - mu_t) / sd_t) * sd_s + mu_s


def fit_pds(train_src: np.ndarray, train_tgt: np.ndarray,
            test_tgt: np.ndarray, window: int = 1,
            alpha: float = 1e-3) -> np.ndarray:
    """Piecewise direct standardisation from target to source coordinates.

    Each source band is regressed on a local target-band window estimated from
    paired training spectra only.  Standardisation keeps the local regressions
    numerically comparable across bands; no test labels are used.
    """
    if train_src.shape != train_tgt.shape:
        raise ValueError("PDS requires paired source and target matrices")
    if window < 0:
        raise ValueError("window must be non-negative")
    mu_s, sd_s = train_src.mean(0), train_src.std(0) + 1e-6
    mu_t, sd_t = train_tgt.mean(0), train_tgt.std(0) + 1e-6
    zt = (train_tgt - mu_t) / sd_t
    zs = (train_src - mu_s) / sd_s
    ztest = (test_tgt - mu_t) / sd_t
    pred = np.empty((len(test_tgt), train_src.shape[1]), dtype=float)
    for j in range(train_src.shape[1]):
        lo, hi = max(0, j - window), min(train_src.shape[1], j + window + 1)
        mapper = Ridge(alpha=alpha, fit_intercept=True)
        mapper.fit(zt[:, lo:hi], zs[:, j])
        pred[:, j] = mapper.predict(ztest[:, lo:hi])
    return pred * sd_s + mu_s


def pair_reconstruction_rmse(train_src: np.ndarray, train_tgt: np.ndarray,
                             alpha: float, groups=None) -> float:
    """Unsupervised paired reconstruction score used for alpha selection."""
    n = len(train_src)
    if groups is None:
        groups = np.arange(n)
    k = min(3, len(np.unique(groups)))
    pred = np.empty_like(train_src)
    for tr, te in GroupKFold(k, shuffle=True, random_state=701).split(train_tgt, groups=groups):
        pred[te] = fit_pair_ridge(train_src[tr], train_tgt[tr], train_tgt[te], alpha)
    return float(np.sqrt(np.mean((pred - train_src) ** 2)))


def select_pair_alpha(train_src: np.ndarray, train_tgt: np.ndarray,
                      groups, candidates=(1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)):
    scores = [(pair_reconstruction_rmse(train_src, train_tgt, a, groups), a)
              for a in candidates]
    return min(scores)[1], min(scores)[0]


def select_pair_alpha_supervised(train_src: np.ndarray, train_tgt: np.ndarray,
                                 y_log: np.ndarray, groups,
                                 candidates=(0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0)):
    """Select translation regularisation using source labels only.

    The validation spectra remain target-camera observations, but their
    hardness labels are the source labels of the same training fruits. No
    outer test label participates in this choice.
    """
    k = min(3, len(np.unique(groups)))
    scores = []
    for alpha in candidates:
        errs = []
        for tr, te in GroupKFold(k, shuffle=True, random_state=907).split(train_src, groups=groups):
            sm = Pipeline([("scale", StandardScaler()),
                           ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                                 max_iter=100000))])
            sm.fit(train_src[tr], y_log[tr])
            mapped = fit_pair_ridge(train_src[tr], train_tgt[tr], train_tgt[te], alpha)
            pred = sm.predict(mapped)
            errs.append(np.mean((pred - y_log[te]) ** 2))
        scores.append((float(np.mean(errs)), alpha))
    return min(scores)[1], min(scores)[0]


def metric(y, p):
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    return {"R2": float(r2_score(y, p)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, p)),
            "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf)}


def paired_table(fruit: str, camera_a: str, camera_b: str):
    z = np.load(ROOT / "baseline_cache" / f"inter_{fruit}_{camera_a}_{camera_b}_overlap.npz")
    A, B = z["Xa"].astype(float), z["Xb"].astype(float)
    d = load_records().df
    p = camera_pairs(d[d.fruit == fruit], camera_a, camera_b)
    # Keep the physical-fruit identifier explicit.  ``uid`` identifies a
    # fruit on one storage day, whereas ``physical_uid`` also joins repeated
    # observations of the same fruit across days within a storage series.
    meta = d[(d.fruit == fruit) & d.firmness_g_cm2.notna()].drop_duplicates("uid")[
        ["uid", "physical_uid", "firmness_g_cm2", "series", "day", "storage_days"]
    ]
    p = p.merge(meta, on="uid", how="inner")
    p["side_rank"] = (p.side != "front").astype(int)
    p = p.sort_values(["uid", "side_rank"]).drop_duplicates("uid").reset_index(drop=True)
    if len(p) != len(A):
        raise ValueError(f"cache/table mismatch: cache={len(A)} table={len(p)}")
    if "uids" not in z.files:
        raise ValueError("paired-spectrum cache has no UID manifest; regenerate it before evaluation")
    cached_uids = z["uids"].astype(str)
    table_uids = p.uid.astype(str).to_numpy()
    if not np.array_equal(cached_uids, table_uids):
        raise ValueError("paired-spectrum cache UID order does not match the paired table")
    return A, B, p


def run(fruit="Avocado", camera_a="VIS", camera_b="VIS_COR",
        repeats=5, folds=5, ridge_alpha=10.0, cache=ROOT / "baseline_cache"):
    A, B, p = paired_table(fruit, camera_a, camera_b)
    y0 = p.firmness_g_cm2.to_numpy(float); y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=folds)):
        pooled = {}
        for direction, src, tgt, src_name, tgt_name in [
                (f"{camera_a}->{camera_b}", A, B, camera_a, camera_b),
                (f"{camera_b}->{camera_a}", B, A, camera_b, camera_a)]:
            for method in ("raw", "scale", "coral", "pair_ridge", "pair_ridge_auto", "pair_ridge_tuned"):
                pred_all = np.empty(len(y0))
                for fold, (tr, te) in enumerate(splits):
                    source_model = Pipeline([
                        ("scale", StandardScaler()),
                        ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                              max_iter=100000)),
                    ])
                    source_model.fit(src[tr], y[tr])
                    chosen_alpha = ridge_alpha
                    train_recon = np.nan
                    if method in ("pair_ridge", "pair_ridge_auto", "pair_ridge_tuned"):
                        if method == "pair_ridge_auto":
                            chosen_alpha, train_recon = select_pair_alpha(src[tr], tgt[tr], groups[tr])
                        elif method == "pair_ridge_tuned":
                            chosen_alpha, train_recon = select_pair_alpha_supervised(src[tr], tgt[tr], y[tr], groups[tr])
                        Xt = fit_pair_ridge(src[tr], tgt[tr], tgt[te], chosen_alpha)
                    else:
                        Xt = align(src[tr], tgt[tr], tgt[te], method)
                    pred = np.expm1(source_model.predict(Xt))
                    pred_all[te] = pred
                    row = metric(y0[te], pred)
                    row.update(direction=direction, method=method, repeat=rep,
                               fold=fold, aggregate="fold", n_test=len(te),
                               ridge_alpha=chosen_alpha, train_recon_rmse=train_recon)
                    rows.append(row)
                row = metric(y0, pred_all)
                row.update(direction=direction, method=method, repeat=rep,
                           fold=-1, aggregate="repeat_pooled", n_test=len(y0),
                           ridge_alpha=ridge_alpha)
                rows.append(row)
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_domain_invariant_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    pooled = out[out["aggregate"] == "repeat_pooled"]
    print(f"{fruit}: paired {camera_a}<->{camera_b}, n={len(p)}")
    print(pooled.groupby(["direction", "method"])[["R2", "RMSE", "MAE", "RPD"]]
          .agg(["mean", "std"]).round(4).to_string())
    print(f"saved: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default="Avocado")
    ap.add_argument("--a", default="VIS")
    ap.add_argument("--b", default="VIS_COR")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.repeats, args.folds,
        args.ridge_alpha, Path(args.cache))
