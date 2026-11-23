"""Paired fruit-level transfer diagnostic for two cameras."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from physfruit.labels import load_records, camera_pairs, CAMERAS
from physfruit.basis import camera_wavelengths
from physfruit.envi import read_cube
from physfruit.realfit import fruit_pixels
from physfruit.evaluation import repeated_group_folds
DATA_ROOT=str(ROOT / "DeepHS-Fruit" / "data")

def mean(f):
    p=os.path.join(DATA_ROOT,f.replace('/',os.sep)).rsplit('.',1)[0]
    c,_=read_cube(p); return c[fruit_pixels(c)].mean(0)

def main(fruit,a,b,repeats):
    tab=load_records(); d=tab.df
    p=camera_pairs(d[d.fruit==fruit],a,b)
    lab=d[(d.fruit==fruit)&d.firmness_g_cm2.notna()].drop_duplicates('uid')[['uid','firmness_g_cm2']]
    p=p.merge(lab,on='uid',how='inner'); p['rank']=(p.side!='front').astype(int); p=p.sort_values(['uid','rank']).drop_duplicates('uid')
    # Use the official wavelength tables, not endpoint interpolation.
    wa=camera_wavelengths(a); wb=camera_wavelengths(b)
    lo=max(wa.min(),wb.min()); hi=min(wa.max(),wb.max()); grid=np.linspace(lo+5,hi-5,20)
    ca=ROOT/'baseline_cache'; ca.mkdir(exist_ok=True); fn=ca/f'inter_{fruit}_{a}_{b}_overlap.npz'
    if fn.exists():
        z=np.load(fn)
        cache_ok = ('uids' in z.files and len(z['Xa']) == len(p)
                    and np.array_equal(z['uids'].astype(str), p.uid.astype(str).to_numpy()))
    else:
        cache_ok = False
    if cache_ok:
        Xa,Xb=z['Xa'],z['Xb']
    else:
        xa=[]; xb=[]
        for i,(_,r) in enumerate(p.iterrows(),1):
            print(f'  reading pair {i}/{len(p)} {r.uid}',flush=True)
            xa.append(np.interp(grid,wa,mean(r['data_file_'+a]))); xb.append(np.interp(grid,wb,mean(r['data_file_'+b])))
        Xa,Xb=np.stack(xa),np.stack(xb)
        # Pandas 3 string arrays may become object arrays; store Unicode so
        # the UID manifest remains readable with np.load(allow_pickle=False).
        np.savez_compressed(fn,Xa=Xa,Xb=Xb,uids=np.asarray(p.uid.to_numpy(),dtype=str))
    y0=p.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=p.uid.to_numpy(); rows=[]
    for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=5)):
        for fold,(tr,te) in enumerate(splits):
            for name,Xtr,Xte,target in [(a,Xa,Xb,b),(b,Xb,Xa,a)]:
                m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]); m.fit(Xtr[tr],y[tr]); pred=np.expm1(m.predict(Xte[te])); rm=np.sqrt(mean_squared_error(y0[te],pred)); rows.append(dict(direction=f'{name}->{target}',R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold))
    out=pd.DataFrame(rows); print(f'\n{fruit}: {len(p)} paired fruits, overlap {lo:.0f}-{hi:.0f} nm ({a}/{b})'); print(out.groupby('direction')[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string()); out.to_csv(ca/f'inter_{fruit}_{a}_{b}_overlap.csv',index=False)

if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--fruit',default='Avocado'); q.add_argument('--a',default='VIS'); q.add_argument('--b',default='VIS_COR'); q.add_argument('--repeats',type=int,default=3); z=q.parse_args(); main(z.fruit,z.a,z.b,z.repeats)
