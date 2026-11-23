"""Repeat-pooled uniform-band robustness controls for the final transfer protocol."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from physfruit.evaluation import repeated_group_folds
from physfruit.inter_alignment import align
from physfruit.inter_domain_invariant import fit_pair_ridge, paired_table

ROOT = Path(__file__).resolve().parent.parent

def score(y_true, pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    return dict(R2=float(r2_score(y_true, pred)), RMSE=rmse,
                MAE=float(mean_absolute_error(y_true, pred)))

def run(fruit, camera_a, camera_b, repeats=5, folds=5, cache=None):
    A, B, p = paired_table(fruit, camera_a, camera_b)
    y0 = p.firmness_g_cm2.to_numpy(float)
    y = np.log1p(y0)
    groups = p.physical_uid.to_numpy()
    rows=[]
    for n_bands in (8,16,32):
        ix=np.linspace(0, A.shape[1]-1, n_bands).round().astype(int)
        for rep, splits in enumerate(repeated_group_folds(groups, repeats, folds)):
            for direction, src_all, tgt_all, target in [
                (f'{camera_a}->{camera_b}', A[:,ix], B[:,ix], camera_b),
                (f'{camera_b}->{camera_a}', B[:,ix], A[:,ix], camera_a),
            ]:
                pred_all={m: np.empty(len(y0)) for m in ('raw','scale','coral','pair_ridge')}
                for tr, te in splits:
                    reg=Pipeline([('scale', StandardScaler()),
                                  ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000))])
                    reg.fit(src_all[tr], y[tr])
                    mapped={
                        'raw': tgt_all[te],
                        'scale': align(src_all[tr], tgt_all[tr], tgt_all[te], 'scale'),
                        'coral': align(src_all[tr], tgt_all[tr], tgt_all[te], 'coral'),
                        'pair_ridge': fit_pair_ridge(src_all[tr], tgt_all[tr], tgt_all[te], alpha=5*n_bands),
                    }
                    for method, Xte in mapped.items():
                        pred_all[method][te]=np.expm1(reg.predict(Xte))
                for method, pred in pred_all.items():
                    r=score(y0,pred)
                    rows.append(dict(fruit=fruit, direction=direction, n_bands=n_bands,
                                     method=method, repeat=rep, aggregate='repeat_pooled', **r))
    out=pd.DataFrame(rows)
    if cache is None: cache=ROOT/'baseline_cache'
    cache=Path(cache); cache.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache/f'inter_band_count_pooled_{fruit}_{camera_a}_{camera_b}.csv', index=False)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--fruit', required=True); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--repeats', type=int, default=5); ap.add_argument('--folds', type=int, default=5)
    args=ap.parse_args()
    out=run(args.fruit,args.a,args.b,args.repeats,args.folds)
    print(out.groupby(['direction','n_bands','method'])[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string())
