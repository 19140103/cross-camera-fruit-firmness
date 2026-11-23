"""Leakage-safe stability-aware spectral interval selection.

For each outer GroupKFold training split, bootstrap fruits, fit a small PLS
model, smooth the absolute coefficient importance over wavelength, and select
four separated four-band intervals. Test folds are never used for selection.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from physfruit.baseline_regression import load_fruit_table
from physfruit.basis import camera_wavelengths


def select_intervals(X, y, wl, rng, n_intervals=4, width=4, n_boot=20,
                     pls_components=6, min_separation=20):
    """Return selected indices and the bootstrap stability score."""
    scores = []
    n = len(X)
    ncomp = max(1, min(pls_components, n - 2, X.shape[1]))
    for _ in range(n_boot):
        ii = rng.integers(0, n, size=n)
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=ncomp, max_iter=1000)),
        ])
        model.fit(X[ii], y[ii])
        coef = np.asarray(model.named_steps["pls"].coef_).reshape(-1)
        # Coefficients are on standardized predictors, so magnitudes are
        # comparable across wavelengths within a bootstrap sample.
        scores.append(np.abs(coef))
    imp = np.asarray(scores)
    # Stability rewards repeatable importance and penalizes unstable peaks.
    stable = imp.mean(0) / (imp.std(0) + 1e-8)
    # Smooth over neighboring bands; hardness information may be a local shape,
    # not a single exact detector channel.
    kernel = np.ones(7, dtype=float) / 7.0
    smooth = np.convolve(stable, kernel, mode="same")

    chosen = []
    remaining = set(np.argsort(smooth)[::-1].tolist())
    half = width // 2
    while remaining and len(chosen) < n_intervals:
        c = max(remaining, key=lambda j: smooth[j])
        lo = max(0, c - half)
        hi = min(len(wl), lo + width)
        lo = max(0, hi - width)
        idx = np.arange(lo, hi)
        if all(abs(float(wl[c] - wl[q])) >= min_separation for q in chosen):
            chosen.append(c)
            # Remove centers whose intervals would overlap or be too close.
            remaining = {j for j in remaining
                         if abs(float(wl[j] - wl[c])) >= min_separation}
        else:
            remaining.discard(c)
    if len(chosen) < n_intervals:
        for c in np.argsort(smooth)[::-1]:
            if all(abs(float(wl[c] - wl[q])) >= min_separation for q in chosen):
                chosen.append(int(c))
            if len(chosen) == n_intervals:
                break
    selected = []
    for c in chosen:
        lo = max(0, c - half)
        hi = min(len(wl), lo + width)
        lo = max(0, hi - width)
        selected.extend(range(lo, hi))
    return np.unique(np.asarray(selected, dtype=int)), smooth


def main(fruit="Avocado", camera="VIS", repeats=5, folds=5, n_boot=20):
    X, d = load_fruit_table(fruit, camera, ROOT / "baseline_cache")
    wl = camera_wavelengths(camera)
    y0 = d.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = d.physical_group.to_numpy()
    rows = []
    selected_all = []
    for rep in range(repeats):
        rng = np.random.default_rng(20261002 + rep)
        ug = np.unique(groups)
        order = rng.permutation(ug)
        rank = {g: i for i, g in enumerate(order)}
        idx = np.argsort([rank[g] for g in groups])
        for fold, (tr0, te0) in enumerate(
                GroupKFold(folds).split(X[idx], y[idx], groups[idx])):
            tr, te = idx[tr0], idx[te0]
            sel, smooth = select_intervals(
                X[tr], y[tr], wl,
                np.random.default_rng(9000 + rep * 100 + fold),
                n_boot=n_boot)
            selected_all.extend(sel.tolist())
            for method, chosen in [("stable_intervals", sel),
                                   ("uniform", np.linspace(0, X.shape[1] - 1,
                                                            len(sel)).round().astype(int)),
                                   ("full", np.arange(X.shape[1]))]:
                m = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=.01, l1_ratio=.5,
                                          max_iter=20000)),
                ])
                m.fit(X[tr][:, chosen], y[tr])
                pred = np.expm1(m.predict(X[te][:, chosen]))
                rm = float(np.sqrt(mean_squared_error(y0[te], pred)))
                rows.append(dict(method=method, n_bands=len(chosen),
                                 R2=float(r2_score(y0[te], pred)), RMSE=rm,
                                 MAE=float(mean_absolute_error(y0[te], pred)),
                                 repeat=rep, fold=fold,
                                 selected=','.join(map(str, chosen))))
    out = pd.DataFrame(rows)
    print(f"\n{fruit}/{camera}: stability-aware interval selection")
    print(out.groupby(["method", "n_bands"])[["R2", "RMSE", "MAE"]]
          .agg(["mean", "std"]).round(4).to_string())
    freq = np.bincount(selected_all, minlength=X.shape[1])
    print("selected-band frequency:",
          [(int(j), round(float(wl[j]), 2), int(freq[j]))
           for j in np.argsort(freq)[-20:][::-1]])
    out.to_csv(ROOT / "baseline_cache" /
               f"intra_{fruit}_{camera}_stable_intervals.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default="Avocado")
    ap.add_argument("--camera", default="VIS")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=20)
    a = ap.parse_args()
    main(a.fruit, a.camera, a.repeats, a.folds, a.bootstrap)
