"""Compare simple paired-camera alignment methods under the same protocol."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from physfruit.labels import load_records

def root_sym(a, eps=1e-5):
    v,U=np.linalg.eigh(a); return (U*np.sqrt(np.maximum(v,eps)))@U.T
def inv_root(a, eps=1e-5):
    v,U=np.linalg.eigh(a); return (U*(1.0/np.sqrt(np.maximum(v,eps))))@U.T
def align(train_src, train_tgt, test_tgt, method):
    ms=train_src.mean(0); mt=train_tgt.mean(0)
    if method=='raw': return test_tgt
    if method=='scale':
        return (test_tgt-mt)/(train_tgt.std(0)+1e-6)*(train_src.std(0)+1e-6)+ms
    if method=='affine':
        out=np.empty_like(test_tgt)
        for j in range(train_src.shape[1]):
            A=np.c_[train_tgt[:,j],np.ones(len(train_tgt))]
            coef=np.linalg.lstsq(A,train_src[:,j],rcond=None)[0]; out[:,j]=test_tgt[:,j]*coef[0]+coef[1]
        return out
    if method=='coral':
        cs=np.cov(train_src-ms,rowvar=False)+1e-4*np.eye(train_src.shape[1]); ct=np.cov(train_tgt-mt,rowvar=False)+1e-4*np.eye(train_tgt.shape[1])
        return (test_tgt-mt)@inv_root(ct)@root_sym(cs)+ms
    raise ValueError(method)

def main(fruit,a,b,repeats):
    z=np.load(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_overlap.npz'); Xa,Xb=z['Xa'],z['Xb']; d=load_records().df
    from physfruit.labels import camera_pairs
    p=camera_pairs(d[d.fruit==fruit],a,b); lab=d[(d.fruit==fruit)&d.firmness_g_cm2.notna()].drop_duplicates('uid')[['uid','firmness_g_cm2']]; p=p.merge(lab,on='uid'); p['rank']=(p.side!='front').astype(int); p=p.sort_values(['uid','rank']).drop_duplicates('uid')
    y0=p.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=p.uid.to_numpy(); rows=[]
    for rep in range(repeats):
        rng=np.random.default_rng(20261002+rep); ug=np.unique(groups); order=rng.permutation(ug); rr={g:i for i,g in enumerate(order)}; idx=np.argsort([rr[g] for g in groups])
        for fold,(tr0,te0) in enumerate(GroupKFold(5).split(Xa[idx],y[idx],groups[idx])):
            tr,te=idx[tr0],idx[te0]
            for name,src,tgt,target in [(a,Xa,Xb,b),(b,Xb,Xa,a)]:
                for method in ['raw','scale','affine','coral']:
                    Xt=align(src[tr],tgt[tr],tgt[te],method)
                    m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]); m.fit(src[tr],y[tr]); pred=np.expm1(m.predict(Xt)); rm=np.sqrt(mean_squared_error(y0[te],pred)); rows.append(dict(direction=f'{name}->{target}',method=method,R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold))
    out=pd.DataFrame(rows); print(f'\n{fruit}: {a}<->{b}, {len(p)} paired fruits'); print(out.groupby(['direction','method'])[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string()); out.to_csv(ROOT/'baseline_cache'/f'inter_{fruit}_{a}_{b}_alignment.csv',index=False)

if __name__=='__main__':
    q=argparse.ArgumentParser(); q.add_argument('--fruit',default='Avocado'); q.add_argument('--a',default='VIS'); q.add_argument('--b',default='VIS_COR'); q.add_argument('--repeats',type=int,default=5); z=q.parse_args(); main(z.fruit,z.a,z.b,z.repeats)
