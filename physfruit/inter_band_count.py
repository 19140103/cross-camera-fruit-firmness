"""Uniform-band compression controls on cached paired spectra."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from physfruit.labels import load_records,camera_pairs
from inter_alignment import align
from inter_domain_invariant import fit_pair_ridge
def main(fruit,a,b,repeats):
 z=np.load(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_overlap.npz'); A,B=z['Xa'],z['Xb']; d=load_records().df; p=camera_pairs(d[d.fruit==fruit],a,b); lab=d[(d.fruit==fruit)&d.firmness_g_cm2.notna()].drop_duplicates('uid')[['uid','firmness_g_cm2']]; p=p.merge(lab,on='uid'); p['rank']=(p.side!='front').astype(int); p=p.sort_values(['uid','rank']).drop_duplicates('uid'); y0=p.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=p.uid.to_numpy(); rows=[]
 for n in [8,16,32]:
  ix=np.linspace(0,A.shape[1]-1,n).round().astype(int)
  for rep in range(repeats):
   rng=np.random.default_rng(20261002+rep); ug=np.unique(groups); order=rng.permutation(ug); rr={g:i for i,g in enumerate(order)}; idx=np.argsort([rr[g] for g in groups])
   for fold,(tr0,te0) in enumerate(GroupKFold(5).split(A[idx],y[idx],groups[idx])):
    tr,te=idx[tr0],idx[te0]
    for name,src,tgt,target in [(a,A[:,ix],B[:,ix],b),(b,B[:,ix],A[:,ix],a)]:
     for method in ['raw','scale','coral','pair_ridge']:
      if method == 'pair_ridge':
       Xt = fit_pair_ridge(src[tr], tgt[tr], tgt[te], alpha=5*n)
      else:
       Xt=align(src[tr],tgt[tr],tgt[te],method)
      m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]); m.fit(src[tr],y[tr]); pred=np.expm1(m.predict(Xt)); rm=np.sqrt(mean_squared_error(y0[te],pred)); rows.append(dict(n_bands=n,direction=f'{name}->{target}',method=method,R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold))
 out=pd.DataFrame(rows); print(f'\n{fruit}: uniform bands {a}<->{b}'); print(out.groupby(['n_bands','direction','method'])[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string()); out.to_csv(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_band_count.csv',index=False)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--fruit',default='Avocado');q.add_argument('--a',default='VIS');q.add_argument('--b',default='VIS_COR');q.add_argument('--repeats',type=int,default=5);z=q.parse_args();main(z.fruit,z.a,z.b,z.repeats)
