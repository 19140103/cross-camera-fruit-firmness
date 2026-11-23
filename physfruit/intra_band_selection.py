"""Leakage-safe intra-camera band-selection baselines."""
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
from physfruit.baseline_regression import load_fruit_table
def corr(a,b):
 a=np.asarray(a)-np.mean(a); b=np.asarray(b)-np.mean(b); den=np.sqrt((a*a).sum()*(b*b).sum()); return float(abs((a*b).sum()/den)) if den>1e-12 else 0.
def main(fruit='Avocado',camera='VIS',repeats=5,folds=5):
 X,d=load_fruit_table(fruit,camera,ROOT/'baseline_cache'); y0=d.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=d.physical_group.to_numpy(); rows=[]
 for rep in range(repeats):
  rng=np.random.default_rng(20261002+rep); ug=np.unique(groups); order=rng.permutation(ug); rr={g:i for i,g in enumerate(order)}; idx=np.argsort([rr[g] for g in groups])
  for fold,(tr0,te0) in enumerate(GroupKFold(folds).split(X[idx],y[idx],groups[idx])):
   tr,te=idx[tr0],idx[te0]; scores=np.array([corr(X[tr,j],y[tr]) for j in range(X.shape[1])]);
   for n in [8,16,32,X.shape[1]]:
    sel=np.argsort(scores)[-n:]
    for method in ['selected','uniform'] if n<X.shape[1] else ['selected']:
     if method=='uniform': sel=np.linspace(0,X.shape[1]-1,n).round().astype(int)
     m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]); m.fit(X[tr][:,sel],y[tr]); pred=np.expm1(m.predict(X[te][:,sel])); rm=np.sqrt(mean_squared_error(y0[te],pred)); rows.append(dict(n_bands=n,method=method,R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold,selected=','.join(map(str,np.sort(sel)))))
 out=pd.DataFrame(rows); print(f'\n{fruit}/{camera}: leakage-safe intra band selection'); print(out.groupby(['n_bands','method'])[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string());
 freq=np.zeros(X.shape[1],int)
 for s in out[(out.method=='selected')&(out.n_bands==8)].selected:
  for j in map(int,s.split(',')):freq[j]+=1
 print('top 8-band selection frequencies:',[(int(j),int(freq[j])) for j in np.argsort(freq)[-16:][::-1]])
 out.to_csv(ROOT/'baseline_cache'/f'intra_{fruit}_{camera}_band_selection.csv',index=False)
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--fruit',default='Avocado');q.add_argument('--camera',default='VIS');q.add_argument('--repeats',type=int,default=5);q.add_argument('--folds',type=int,default=5);z=q.parse_args();main(z.fruit,z.camera,z.repeats,z.folds)
