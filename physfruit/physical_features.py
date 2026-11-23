"""Extract and evaluate PROSPECT-derived physical features on DeepHS-Fruit.

The inversion is vectorised over fruit spectra and random initialisations.  A
fruit-level median over initialisations is used as the feature value; the
within-fruit spread is retained as a numerical stability diagnostic.  This
module deliberately keeps the downstream model simple (the same scaled
ElasticNet used by the spectral baseline) so that any gain can be attributed
to the representation rather than a larger regressor.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from physfruit.baseline_regression import load_fruit_table
from physfruit.basis import build_basis
from physfruit.forward import prospect_RT
from physfruit.evaluation import repeated_group_folds

DT = torch.float64
ORDER = ["cab", "car", "anth", "brown", "water", "dm"]


def _init_raw_n(n: int, seeds: int) -> torch.Tensor:
    # N = 1 + softplus(raw_N); initial N values are around 2--3.
    g = torch.Generator().manual_seed(20261002)
    return (torch.randn(n, seeds, 1, generator=g, dtype=DT) * 0.5 + 0.5)


def invert_batch(X: np.ndarray, basis, seeds: int = 3, steps: int = 1000,
                lr: float = 0.03) -> dict[str, np.ndarray]:
    """Fit every spectrum with several random starts in one torch graph.

    Returns arrays with shape (n_fruits, seeds, ...), plus the objective RMSE.
    The observations are repeated along the seed dimension, while each start
    has an independent set of learnable parameters.
    """
    n, bands = X.shape
    k = len(ORDER)
    b = seeds
    eps = torch.tensor(basis.eps, dtype=DT)
    nr = torch.tensor(basis.n_refr, dtype=DT)
    y = torch.tensor(X, dtype=DT)[:, None, :].expand(n, b, bands)

    raw_n = _init_raw_n(n, b).requires_grad_(True)
    # q=c/N is the per-layer absorption.  Starting near one is intentionally
    # weakly informative; the fit must be driven by the measured spectrum.
    g = torch.Generator().manual_seed(20261003)
    log_q = (torch.randn(n, b, k, generator=g, dtype=DT) * 0.8).requires_grad_(True)
    opt = torch.optim.Adam([raw_n, log_q], lr=lr)
    for step in range(steps):
        opt.zero_grad()
        N = 1.0 + torch.nn.functional.softplus(raw_n)
        c = N * log_q.exp()
        R, _ = prospect_RT(N, c, eps, nr)
        loss = ((R - y) ** 2).mean()
        loss.backward()
        opt.step()
        # A small progress line helps long runs remain observable in a shell.
        if step in (0, steps - 1):
            print(f"  inversion step {step + 1}/{steps}, loss={float(loss):.6g}", flush=True)

    with torch.no_grad():
        N = 1.0 + torch.nn.functional.softplus(raw_n)
        q = log_q.exp()
        c = N * q
        R, _ = prospect_RT(N, c, eps, nr)
        resid = R - y
        rmse = torch.sqrt((resid ** 2).mean(dim=-1))
    return {"N": N.cpu().numpy(), "q": q.cpu().numpy(),
            "c": c.cpu().numpy(), "rmse": rmse.cpu().numpy(),
            "fit": R.cpu().numpy()}


def summarise_features(inv: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return median features and per-fruit seed stability diagnostics."""
    N, q, rmse = np.squeeze(inv["N"], axis=-1), inv["q"], inv["rmse"]
    # Relative range is robust to the absolute scale of a parameter.
    def spread(a):
        return 100.0 * (np.max(a, axis=1) - np.min(a, axis=1)) / np.maximum(1e-12, np.abs(np.mean(a, axis=1)))

    med_q = np.median(q, axis=1)
    feat = pd.DataFrame({"N": np.median(N, axis=1),
                         "cab_per_N": med_q[:, 0], "car_per_N": med_q[:, 1],
                         "anth_per_N": med_q[:, 2], "brown_per_N": med_q[:, 3],
                         "water_per_N": med_q[:, 4], "dm_per_N": med_q[:, 5],
                         "cab_car_ratio": med_q[:, 0] / np.maximum(med_q[:, 1], 1e-12),
                         "cab_fraction": med_q[:, 0] / np.maximum(med_q[:, :3].sum(axis=1), 1e-12),
                         "rmse": np.median(rmse, axis=1)})
    stab = pd.DataFrame({"N_spread_pct": spread(N),
                         "cab_per_N_spread_pct": spread(q[:, :, 0]),
                         "car_per_N_spread_pct": spread(q[:, :, 1]),
                         "cab_car_ratio_spread_pct": spread(q[:, :, 0] / np.maximum(q[:, :, 1], 1e-12)),
                         "rmse_min": np.min(rmse, axis=1),
                         "rmse_max": np.max(rmse, axis=1)})
    return feat, stab


def metrics(y, pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return {"R2": float(r2_score(y, pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y, pred)),
            "RPD": float(np.std(y, ddof=1) / rmse if rmse else np.inf)}


def evaluate(X_spec: np.ndarray, X_phys: np.ndarray, d: pd.DataFrame,
             repeats: int = 5, folds: int = 5, log_target: bool = True) -> pd.DataFrame:
    """Fruit-group CV for spectra, physical features, and their concatenation."""
    y0 = d.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0) if log_target else y0
    groups = d.physical_group.to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    all_rows = []
    banks = {"spectral": X_spec, "physical": X_phys,
             "spectral+physical": np.column_stack([X_spec, X_phys])}
    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=n_splits)):
        for fold, (tr, te) in enumerate(splits):
            for name, X in banks.items():
                model = Pipeline([("scale", StandardScaler()),
                                  ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000))])
                model.fit(X[tr], y[tr])
                pred = np.asarray(model.predict(X[te])).reshape(-1)
                if log_target:
                    pred = np.expm1(pred)
                row = metrics(y0[te], pred)
                row.update(model=name, repeat=rep, fold=fold, n_test=len(te))
                all_rows.append(row)
    return pd.DataFrame(all_rows)


def run(fruit: str = "Avocado", camera: str = "VIS", seeds: int = 3,
        steps: int = 1000, repeats: int = 5, folds: int = 5,
        cache: Path = Path("baseline_cache")) -> None:
    X, d = load_fruit_table(fruit, camera, cache)
    basis = build_basis(camera)
    print(f"{fruit}/{camera}: fitting {len(X)} spectra, {seeds} starts, {steps} steps")
    inv = invert_batch(X, basis, seeds=seeds, steps=steps)
    feat, stab = summarise_features(inv)
    outdir = Path(cache)
    outdir.mkdir(parents=True, exist_ok=True)
    feat_path = outdir / f"prospect_features_{fruit}_{camera}.csv"
    stab_path = outdir / f"prospect_stability_{fruit}_{camera}.csv"
    feat.to_csv(feat_path, index=False)
    stab.to_csv(stab_path, index=False)
    print("feature summary:")
    print(feat.describe().T.round(5).to_string())
    print("stability median (% relative range):")
    print(stab[[c for c in stab if c.endswith("spread_pct")]].median().round(3).to_string())
    # Add log-transformed physical quantities so a linear model does not have
    # to absorb orders of magnitude in the raw positive estimates.
    phys_cols = list(feat.columns)
    Xp = feat[phys_cols].to_numpy(float)
    Xp = np.sign(Xp) * np.log1p(np.abs(Xp))
    result = evaluate(X, Xp, d, repeats=repeats, folds=folds, log_target=True)
    print("\nRepeated fruit-group CV (original g/cm2 scale):")
    print(result.groupby("model")[["R2", "RMSE", "MAE", "RPD"]].agg(["mean", "std"]).round(4).to_string())
    result.to_csv(outdir / f"prospect_regression_{fruit}_{camera}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default="Avocado")
    ap.add_argument("--camera", default="VIS")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", default="baseline_cache")
    a = ap.parse_args()
    run(a.fruit, a.camera, a.seeds, a.steps, a.repeats, a.folds, Path(a.cache))
