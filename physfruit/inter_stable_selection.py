"""Training-fold-only stable band selection for paired camera transfer."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from physfruit.labels import load_records,camera_pairs,CAMERAS
from inter_alignment import align
def corr(a,b):
 a=np.asarray(a); b=np.asarray(b); a=a-a.mean(); b=b-b.mean(); den=np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/den) if den>1e-12 else 0.
def main(fruit,a,b,repeats,n):
 z=np.load(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_overlap.npz'); A,B=z['Xa'],z['Xb']; d=load_records().df; p=camera_pairs(d[d.fruit==fruit],a,b); lab=d[(d.fruit==fruit)&d.firmness_g_cm2.notna()].drop_duplicates('uid')[['uid','firmness_g_cm2']]; p=p.merge(lab,on='uid'); p['rank']=(p.side!='front').astype(int); p=p.sort_values(['uid','rank']).drop_duplicates('uid'); y0=p.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=p.uid.to_numpy(); rows=[]; picked=[]
 for rep in range(repeats):
  rng=np.random.default_rng(20261002+rep); ug=np.unique(groups); order=rng.permutation(ug); rr={g:i for i,g in enumerate(order)}; idx=np.argsort([rr[g] for g in groups])
  for fold,(tr0,te0) in enumerate(GroupKFold(5).split(A[idx],y[idx],groups[idx])):
   tr,te=idx[tr0],idx[te0]
   # Selection uses paired training spectra and training labels only.
   score=[]
   for j in range(A.shape[1]):
    stability=max(0.,corr(A[tr,j],B[tr,j])); relevance=abs(corr(A[tr,j],y[tr]));
    md=abs(A[tr,j].mean()-B[tr,j].mean())/(A[tr,j].std()+B[tr,j].std()+1e-6)
    score.append(stability*relevance/(1.+md))
   sel=np.argsort(score)[-n:]; sel=np.sort(sel); picked.extend(sel.tolist())
   for name,src,tgt,target in [(a,A,B,b),(b,B,A,a)]:
    for method in ['raw','scale','coral']:
     Xt=align(src[tr][:,sel],tgt[tr][:,sel],tgt[te][:,sel],method); m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]); m.fit(src[tr][:,sel],y[tr]); pred=np.expm1(m.predict(Xt)); rm=np.sqrt(mean_squared_error(y0[te],pred)); rows.append(dict(n_bands=n,direction=f'{name}->{target}',method=method,R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold))
 out=pd.DataFrame(rows); print(f'\n{fruit}: train-fold stable selection ({n} bands), {a}<->{b}'); print(out.groupby(['direction','method'])[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string()); print('selection frequency over common-grid positions:',np.bincount(picked,minlength=A.shape[1]).tolist()); out.to_csv(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_stable_selection.csv',index=False)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--fruit',default='Avocado');q.add_argument('--a',default='VIS');q.add_argument('--b',default='VIS_COR');q.add_argument('--repeats',type=int,default=5);q.add_argument('--n',type=int,default=8);z=q.parse_args();main(z.fruit,z.a,z.b,z.repeats,z.n)
