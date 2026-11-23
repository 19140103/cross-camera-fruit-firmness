"""Fruit-level preliminary regression baselines for continuous firmness.

This script intentionally uses repeated GroupKFold rather than the released
record-level split.  The public annotations can assign front/back records of
one physical fruit to different split names; uid-level grouping prevents this
from inflating test performance.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physfruit.envi import read_cube  # noqa: E402
from physfruit.labels import load_records, CUBE_ROOT  # noqa: E402
from physfruit.realfit import fruit_pixels  # noqa: E402
from physfruit.evaluation import repeated_group_folds  # noqa: E402


def spectrum(data_file: str) -> np.ndarray:
    root = os.path.join(CUBE_ROOT, data_file.replace("/", os.sep))
    cube, _ = read_cube(root.rsplit(".", 1)[0])
    m = fruit_pixels(cube)
    return np.asarray(cube[m].mean(axis=0), dtype=np.float32)


def load_fruit_table(fruit: str, camera: str, cache: Path) -> tuple[np.ndarray, pd.DataFrame]:
    tab = load_records()
    d = tab.df[(tab.df.fruit == fruit) & (tab.df.camera == camera)
               & tab.df.firmness_g_cm2.notna() & tab.df.cube_exists].copy()
    # One view per fruit/day record. Prefer front when both views are available.
    # The CV group below is stricter than uid: M2 contains a few fruit IDs
    # observed on multiple storage days, so those longitudinal records must
    # stay in one fold.
    d["side_rank"] = (d.side != "front").astype(int)
    d = d.sort_values(["uid", "side_rank"]).drop_duplicates("uid").copy()
    d["physical_group"] = d.fruit + "|" + d.series + "|" + d.fruit_id
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / f"{fruit}_{camera}_firmness.npz"
    if key.exists():
        z = np.load(key, allow_pickle=False)
        X = z["X"].astype(np.float32)
        # Keep the current annotations as source of labels and grouping.
        if X.shape[0] != len(d):
            key.unlink()
        else:
            return X, d.reset_index(drop=True)
    rows = []
    for i, (_, row) in enumerate(d.iterrows(), 1):
        print(f"  reading {i:3d}/{len(d)} {row.uid}", flush=True)
        rows.append(spectrum(row.data_file))
    X = np.stack(rows)
    np.savez_compressed(key, X=X)
    return X, d.reset_index(drop=True)


def model_bank(n_components: int) -> dict:
    return {
        f"PLSR-{n_components}": Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=n_components, max_iter=1000)),
        ]),
        "SVR-rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(C=10.0, epsilon=0.1, gamma="scale")),
        ]),
        "ElasticNet": Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
        ]),
        "RF": RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                     min_samples_leaf=2, random_state=0, n_jobs=-1),
    }


def metrics(y, pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return dict(R2=float(r2_score(y, pred)), RMSE=rmse,
                MAE=float(mean_absolute_error(y, pred)),
                RPD=float(np.std(y, ddof=1) / rmse if rmse else np.inf))


def run(fruit: str, camera: str, repeats: int, folds: int, log_target: bool,
        n_components: int, cache: Path) -> None:
    X, d = load_fruit_table(fruit, camera, cache)
    y0 = d.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0) if log_target else y0
    groups = d.physical_group.to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
    all_rows = []
    print(f"\n{fruit} / {camera}: {len(d)} fruit-day records, "
          f"{d.physical_group.nunique()} physical groups, "
          f"target={'log1p' if log_target else 'raw'}")
    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=n_splits)):
        pooled_pred = {name: np.empty(len(y0)) for name in model_bank(n_components)}
        for fold, (tr, te) in enumerate(splits):
            bank = model_bank(n_components)
            for name, model in bank.items():
                model.fit(X[tr], y[tr])
                p = np.asarray(model.predict(X[te])).reshape(-1)
                if log_target:
                    p = np.expm1(p)
                mm = metrics(y0[te], p)
                pooled_pred[name][te] = p
                mm.update(model=name, repeat=rep, fold=fold, aggregate="fold", n_test=len(te))
                all_rows.append(mm)
        for name, p in pooled_pred.items():
            mm = metrics(y0, p)
            mm.update(model=name, repeat=rep, fold=-1, aggregate="repeat_pooled", n_test=len(y0))
            all_rows.append(mm)
    out = pd.DataFrame(all_rows)
    print("\nRepeated GroupKFold summary (fold-level; metrics on original g/cm^2 scale)")
    print(out[out["aggregate"] == "fold"].groupby("model")[['R2','RMSE','MAE','RPD']].agg(['mean','std']).round(4).to_string())
    print("\nRepeated GroupKFold summary (repeat-pooled; one score per group partition)")
    print(out[out["aggregate"] == "repeat_pooled"].groupby("model")[['R2','RMSE','MAE','RPD']].agg(['mean','std']).round(4).to_string())
    out_path = cache / f"baseline_{fruit}_{camera}_{'log' if log_target else 'raw'}.csv"
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default="Avocado")
    ap.add_argument("--camera", default="VIS")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--log-target", action="store_true")
    ap.add_argument("--components", type=int, default=8)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    a = ap.parse_args()
    run(a.fruit, a.camera, a.repeats, a.folds, a.log_target, a.components, Path(a.cache))
