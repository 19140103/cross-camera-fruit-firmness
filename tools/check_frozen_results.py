"""Check archived CSV integrity and main summaries without fitting models."""
from pathlib import Path
import csv
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "baseline_cache"


def main():
    with (ROOT / "docs/frozen_results_manifest.csv").open(encoding="utf-8") as f:
        files = list(csv.DictReader(f))
    for row in files:
        path = ROOT / row["path"]
        assert path.stat().st_size == int(row["bytes"]), row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]

    keys = ["fruit", "direction", "method"]
    frozen = pd.read_csv(CACHE / "main_results_frozen_2026-10-03.csv", float_precision="round_trip")
    assert not frozen.duplicated(keys).any()
    frozen = frozen.set_index(keys).sort_index()
    paths = sorted(CACHE.glob("inter_transfer_baselines_*.csv"))
    assert len(paths) == 6
    cached = pd.concat([pd.read_csv(p, float_precision="round_trip") for p in paths])
    cached = cached[cached["aggregate"].eq("repeat_pooled")]
    assert not cached.duplicated([*keys, "repeat"]).any()
    assert cached.groupby(keys).size().eq(5).all()
    summary = cached.groupby(keys)[["R2", "RMSE", "MAE"]].agg(["mean", "std"])
    summary.columns = [f"{metric}_{'sd' if stat == 'std' else stat}" for metric, stat in summary.columns]
    assert summary.index.equals(frozen.index)
    np.testing.assert_allclose(summary, frozen[summary.columns], rtol=1e-12, atol=1e-10)

    scores = frozen.R2_mean.unstack("method")
    assert scores.shape == (12, 6) and np.isfinite(scores).all().all()
    improved = int((scores.pair_ridge > scores.raw).sum())
    positive = int((scores.pair_ridge > 0).sum())
    best = scores.index[scores.pair_ridge > scores.drop(columns="pair_ridge").max(axis=1)].tolist()
    assert (improved, positive, len(best)) == (10, 6, 4)
    print(f"Verified {len(files)} frozen CSV hashes and all main R2/RMSE/MAE means and SDs.")
    print(f"Pair Ridge: {improved}/12 improved versus Raw; {positive}/12 positive; {len(best)}/12 strict highest means.")
    for fruit, direction in best:
        print(f"  {fruit}: {direction}")


if __name__ == "__main__":
    main()
