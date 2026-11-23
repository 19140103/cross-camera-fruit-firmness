"""Forward storage-time holdout for postharvest applicability.

The earliest ~60% of observed storage days estimate the source hardness model
and the paired spectral translation; later storage days are held out as a
future postharvest condition.  This is a single chronological stress test per
fruit, not a replacement for the repeated fruit-level main protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
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
        "RPD": float(np.std(y, ddof=1) / rmse if len(y) > 1 and rmse else np.nan),
    }


def run(fruit: str, camera_a: str, camera_b: str,
        train_fraction: float = 0.6,
        ridge_alpha: float | None = None,
        cache: Path = ROOT / "baseline_cache") -> pd.DataFrame:
    A, B, p = paired_table(fruit, camera_a, camera_b)
    days = np.sort(p.storage_days.astype(float).unique())
    if len(days) < 4:
        raise ValueError(f"{fruit} has too few storage days for forward holdout")
    n_train_days = max(2, int(np.ceil(len(days) * train_fraction)))
    n_train_days = min(n_train_days, len(days) - 1)
    cutoff = float(days[n_train_days - 1])
    tr = np.flatnonzero(p.storage_days.to_numpy(float) <= cutoff)
    te = np.flatnonzero(p.storage_days.to_numpy(float) > cutoff)
    if len(tr) < 4 or len(te) < 2:
        raise ValueError(f"invalid storage split for {fruit}: train={len(tr)}, test={len(te)}")
    train_uids = set(p.iloc[tr].physical_uid)
    test_uids = set(p.iloc[te].physical_uid)
    if not train_uids.isdisjoint(test_uids):
        raise AssertionError("A physical fruit appears in both chronological train and test sets")
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = p[["uid", "physical_uid", "storage_days"]].copy()
    manifest.insert(0, "paired_row", np.arange(len(p)))
    manifest["split"] = np.where(manifest.storage_days <= cutoff, "train", "test")
    manifest["cutoff_storage_day"] = cutoff
    manifest_dir = cache / "uid_manifests"
    manifest_dir.mkdir(exist_ok=True)
    manifest.to_csv(manifest_dir / f"inter_storage_holdout_{fruit}_{camera_a}_{camera_b}_uids.csv",
                    index=False)
    alpha = float(ridge_alpha if ridge_alpha is not None else 5.0 * A.shape[1])
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    rows: list[dict] = []
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
        mapped = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha)
        Xs = {
            "raw": tgt[te],
            "scale": align(src[tr], tgt[tr], tgt[te], "scale"),
            "coral": align(src[tr], tgt[tr], tgt[te], "coral"),
            "pair_ridge": mapped,
        }
        for method, Xte in Xs.items():
            pred = np.expm1(reg.predict(Xte))
            row = metric(y0[te], pred)
            row.update(
                fruit=fruit, direction=direction, method=method,
                aggregate="forward_storage_holdout", n_train=len(tr), n_test=len(te),
                train_days="|".join(str(x) for x in days[:n_train_days]),
                test_days="|".join(str(x) for x in days[n_train_days:]),
                cutoff_storage_day=cutoff, n_storage_days=len(days),
                n_common_bands=A.shape[1], alpha=alpha,
                source_camera=src_name, target_camera=tgt_name,
            )
            rows.append(row)
        # A time-only control asks whether the spectral models add information
        # beyond the known storage-day trend in this forward split.  It uses
        # early training labels only and is not a camera-transfer method.
        day_model = Ridge(alpha=1.0)
        day_model.fit(p.storage_days.to_numpy(float)[tr, None], y[tr])
        day_pred = np.expm1(day_model.predict(p.storage_days.to_numpy(float)[te, None]))
        row = metric(y0[te], day_pred)
        row.update(
            fruit=fruit, direction="time_only", method="storage_day_ridge",
            aggregate="forward_storage_holdout", n_train=len(tr), n_test=len(te),
            train_days="|".join(str(x) for x in days[:n_train_days]),
            test_days="|".join(str(x) for x in days[n_train_days:]),
            cutoff_storage_day=cutoff, n_storage_days=len(days),
            n_common_bands=A.shape[1], alpha=np.nan,
            source_camera="metadata", target_camera="metadata",
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"inter_storage_holdout_{fruit}_{camera_a}_{camera_b}.csv"
    out.to_csv(path, index=False)
    print(f"{fruit}: {camera_a}<->{camera_b}, train_days={days[:n_train_days].tolist()}, "
          f"test_days={days[n_train_days:].tolist()}, saved={path}")
    print(out.groupby(["direction", "method"])[["R2", "RMSE", "MAE", "RPD"]]
          .mean().round(4).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--train-fraction", type=float, default=0.6)
    ap.add_argument("--ridge-alpha", type=float, default=None)
    ap.add_argument("--cache", default=str(ROOT / "baseline_cache"))
    args = ap.parse_args()
    run(args.fruit, args.a, args.b, args.train_fraction, args.ridge_alpha,
        Path(args.cache))
