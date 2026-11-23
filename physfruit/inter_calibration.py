"""Simple paired-camera calibration controls for the transfer diagnostic."""
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
from physfruit.labels import load_records,camera_pairs,CAMERAS,CUBE_ROOT
from physfruit.envi import read_cube
from physfruit.realfit import fruit_pixels
from physfruit.evaluation import repeated_group_folds
def read(f):
 p=os.path.join(CUBE_ROOT,f.replace('/',os.sep)).rsplit('.',1)[0]; c,_=read_cube(p); return c[fruit_pixels(c)].mean(0)
def main(fruit,a,b,repeats):
 d=load_records().df; p=camera_pairs(d[d.fruit==fruit],a,b); lab=d[(d.fruit==fruit)&d.firmness_g_cm2.notna()].drop_duplicates('uid')[['uid','firmness_g_cm2']]; p=p.merge(lab,on='uid'); p['rank']=(p.side!='front').astype(int); p=p.sort_values(['uid','rank']).drop_duplicates('uid')
 wa=np.linspace(CAMERAS[a]['lo'],CAMERAS[a]['hi'],CAMERAS[a]['n_bands']); wb=np.linspace(CAMERAS[b]['lo'],CAMERAS[b]['hi'],CAMERAS[b]['n_bands']); lo=max(wa.min(),wb.min()); hi=min(wa.max(),wb.max()); grid=np.linspace(lo+5,hi-5,20); cp=ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_overlap.npz'; z=np.load(cp); Xa,Xb=z['Xa'],z['Xb']; y0=p.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=p.uid.to_numpy(); rows=[]
 for rep, splits in enumerate(repeated_group_folds(groups, repeats=repeats, folds=5)):
  for fold,(tr,te) in enumerate(splits):
   for name,src,tgt,target in [(a,Xa,Xb,b),(b,Xb,Xa,a)]:
    # Paired affine diagonal calibration learned only on training fruits.
    mu_s,sd_s=src[tr].mean(0),src[tr].std(0)+1e-6; mu_t,sd_t=tgt[tr].mean(0),tgt[tr].std(0)+1e-6
    zt=(tgt[te]-mu_t)/sd_t*sd_s+mu_s
    m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]); m.fit(src[tr],y[tr]); pred=np.expm1(m.predict(zt)); rm=np.sqrt(mean_squared_error(y0[te],pred)); rows.append(dict(direction=f'{name}->{target} calibrated',R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold))
 out=pd.DataFrame(rows); print(f'\n{fruit}: paired calibration {a}<->{b}, {len(p)} fruits'); print(out.groupby('direction')[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string()); out.to_csv(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_calibrated.csv',index=False)
if __name__=='__main__':
 q=argparse.ArgumentParser(); q.add_argument('--fruit',default='Avocado'); q.add_argument('--a',default='VIS'); q.add_argument('--b',default='VIS_COR'); q.add_argument('--repeats',type=int,default=3); z=q.parse_args(); main(z.fruit,z.a,z.b,z.repeats)
