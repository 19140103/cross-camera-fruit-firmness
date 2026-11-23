"""Unsupervised soft region pooling controls for fruit-level spectra.

The public dataset supplies fruit-level firmness only, so a region module must
not pretend to learn a pixel-level firmness map.  This script tests a weaker,
auditable hypothesis inspired by prompt-free region proposal methods: pixels
with smooth, non-saturated spectra and moderate brightness are more reliable
for estimating the fruit spectrum than specular or mixed pixels.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.baseline_regression import load_fruit_table
from physfruit.envi import read_cube
from physfruit.labels import CUBE_ROOT
from physfruit.realfit import fruit_pixels
from physfruit.evaluation import repeated_group_folds


def _quality_weights(px: np.ndarray) -> np.ndarray:
    """Return positive pixel weights from spectral quality only.

    Brightness is trimmed by the existing fruit mask.  The remaining score
    penalises spectral outliers and high second-difference noise.  All
    normalisations are within one cube, so no label or test-set statistics are
    used.
    """
    b = px.mean(axis=1)
    med = np.median(px, axis=0)
    scale = np.median(np.abs(px - med), axis=0) + 1e-4
    dist = np.sqrt(np.mean(((px - med) / scale) ** 2, axis=1))
    d2 = np.diff(px, n=2, axis=1)
    rough = np.median(np.abs(d2), axis=1)
    # Robust z scores, clipped so a few severe outliers cannot remove all
    # spatial coverage.
    def rz(v):
        m = np.median(v); s = np.median(np.abs(v - m)) + 1e-9
        return np.clip((v - m) / (1.4826 * s), -4.0, 4.0)
    score = 0.6 * rz(dist) + 0.4 * rz(rough) + 0.1 * np.abs(rz(b))
    return np.exp(-0.5 * score)


def pooled_spectra(data_file: str) -> dict[str, np.ndarray]:
    stem = os.path.join(CUBE_ROOT, data_file.replace("/", os.sep))
    cube, _ = read_cube(stem.rsplit(".", 1)[0])
    m = fruit_pixels(cube)
    px = cube[m].astype(np.float64)
    w = _quality_weights(px)
    w /= w.sum()
    return {
        "mean": px.mean(axis=0),
        "median": np.median(px, axis=0),
        "quality_weighted": (px * w[:, None]).sum(axis=0),
        "trimmed_mean": px[np.argsort(np.abs(px - np.median(px, axis=0)).mean(axis=1))[:max(10, int(.7 * len(px)))]].mean(axis=0),
    }


def load_views(fruit: str, camera: str, cache: Path):
    tab = load_fruit_table(fruit, camera, cache)[1]
    # One view per physical fruit record, matching the spectral baseline.
    rows = []
    for i, (_, row) in enumerate(tab.iterrows(), 1):
        print(f"  pooling {i:3d}/{len(tab)} {row.uid}", flush=True)
        z = pooled_spectra(row.data_file)
        rows.append(z)
    return tab.reset_index(drop=True), {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def metrics(y, pred):
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"R2": float(r2_score(y, pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, pred)),
            "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf)}


def evaluate(Xs, d, repeats=5, folds=5):
    y0 = d.firmness_g_cm2.to_numpy(float); y = np.log1p(y0)
    groups = d.physical_group.to_numpy(); n_splits = min(folds, len(np.unique(groups)))
    rows = []
    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=n_splits)):
        for fold, (tr, te) in enumerate(splits):
            for name, X in Xs.items():
                model = Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=.01, l1_ratio=.5, max_iter=20000))])
                model.fit(X[tr], y[tr]); p = np.expm1(np.asarray(model.predict(X[te])).reshape(-1))
                row = metrics(y0[te], p); row.update(pool=name, repeat=rep, fold=fold, n_test=len(te)); rows.append(row)
    return pd.DataFrame(rows)


def run(fruit="Avocado", camera="VIS", repeats=5, folds=5, cache=Path("baseline_cache")):
    d, Xs = load_views(fruit, camera, cache)
    out = evaluate(Xs, d, repeats, folds)
    cache.mkdir(parents=True, exist_ok=True)
    for name, X in Xs.items(): np.savez_compressed(cache / f"region_pool_{fruit}_{camera}_{name}.npz", X=X)
    out.to_csv(cache / f"region_pool_{fruit}_{camera}.csv", index=False)
    print(out.groupby("pool")[["R2", "RMSE", "MAE", "RPD"]].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fruit", default="Avocado"); ap.add_argument("--camera", default="VIS"); ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--cache", default="baseline_cache"); a = ap.parse_args()
    run(a.fruit, a.camera, a.repeats, a.folds, Path(a.cache))
