"""Backward-compatible wrapper for the official-wavelength transfer check.

Use :mod:`inter_transfer` for new runs. The original implementation used
endpoint-linear wavelength grids and is retained only so old commands do not
silently produce publishable-looking numbers.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from physfruit.labels import load_records, camera_pairs  # noqa: E402
from physfruit.envi import read_cube  # noqa: E402
from physfruit.realfit import fruit_pixels  # noqa: E402
from physfruit.labels import CAMERAS  # noqa: E402

DATA_ROOT=str(ROOT / "DeepHS-Fruit" / "data")

def read_mean(f):
    p=os.path.join(DATA_ROOT,f.replace('/',os.sep)).rsplit('.',1)[0]
    c,_=read_cube(p); return c[fruit_pixels(c)].mean(0)

def score(y,p):
    rm=float(np.sqrt(mean_squared_error(y,p)))
    return dict(R2=r2_score(y,p),RMSE=rm,MAE=mean_absolute_error(y,p))

def run(fruit='Avocado', repeats=3):
    tab=load_records(); d=tab.df
    pairs=camera_pairs(d[d.fruit==fruit], 'VIS','NIR')
    # Keep one side per uid; front is preferred. Attach one nonzero firmness label.
    lab=d[(d.fruit==fruit)&d.firmness_g_cm2.notna()].drop_duplicates('uid')[['uid','firmness_g_cm2']]
    pairs=pairs.merge(lab,on='uid',how='inner')
    pairs['rank']=(pairs.side!='front').astype(int)
    pairs=pairs.sort_values(['uid','rank']).drop_duplicates('uid')
    wl_v=np.linspace(CAMERAS['VIS']['lo'],CAMERAS['VIS']['hi'],CAMERAS['VIS']['n_bands'])
    wl_n=np.linspace(CAMERAS['NIR']['lo'],CAMERAS['NIR']['hi'],CAMERAS['NIR']['n_bands'])
    grid=np.linspace(925,995,20)
    Xv=[]; Xn=[]
    cache=ROOT/'baseline_cache'; cache.mkdir(exist_ok=True)
    cp=cache/f'inter_{fruit}_overlap.npz'
    if cp.exists() and len(np.load(cp)['Xv'])==len(pairs):
        z=np.load(cp); Xv=z['Xv']; Xn=z['Xn']
    else:
        for i,(_,r) in enumerate(pairs.iterrows(),1):
            print(f'  reading pair {i}/{len(pairs)} {r.uid}',flush=True)
            Xv.append(np.interp(grid,wl_v,read_mean(r.data_file_VIS)))
            Xn.append(np.interp(grid,wl_n,read_mean(r.data_file_NIR)))
        Xv=np.stack(Xv); Xn=np.stack(Xn); np.savez_compressed(cp,Xv=Xv,Xn=Xn)
    y0=pairs.firmness_g_cm2.to_numpy(float); y=np.log1p(y0)
    groups=pairs.uid.to_numpy(); rows=[]
    for rep in range(repeats):
        rng=np.random.default_rng(20261002+rep); ug=np.unique(groups); order=rng.permutation(ug); rank={g:i for i,g in enumerate(order)}
        idx=np.argsort([rank[g] for g in groups]); sp=GroupKFold(5)
        for fold,(a,b) in enumerate(sp.split(Xv[idx],y[idx],groups[idx])):
            tr,te=idx[a],idx[b]
            for train_name,Xtr,Xte in [('VIS',Xv,Xn),('NIR',Xn,Xv)]:
                m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))])
                m.fit(Xtr[tr],y[tr]); p=np.expm1(m.predict(Xte[te])); q=score(y0[te],p); q.update(direction=f'{train_name}->'+('NIR' if train_name=='VIS' else 'VIS'),repeat=rep,fold=fold,n=len(te)); rows.append(q)
    out=pd.DataFrame(rows); print(f'\n{fruit}: {len(pairs)} paired fruit, overlap 925-995 nm')
    print(out.groupby('direction')[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string())
    out.to_csv(cache/f'inter_{fruit}_overlap.csv',index=False)

if __name__=='__main__':
    print('Deprecated: run physfruit/inter_transfer.py instead; it uses official wavelength tables.')
