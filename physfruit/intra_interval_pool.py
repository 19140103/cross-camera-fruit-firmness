"""Spectral interval pooling controls for continuous firmness."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from physfruit.baseline_regression import load_fruit_table
def pool(X,n):
    edges=np.linspace(0,X.shape[1],n+1).round().astype(int); return np.stack([X[:,edges[i]:edges[i+1]].mean(1) for i in range(n)],1)
def main(fruit,camera,repeats,folds):
    X,d=load_fruit_table(fruit,camera,ROOT/'baseline_cache'); y0=d.firmness_g_cm2.to_numpy(float); y=np.log1p(y0); groups=d.physical_group.to_numpy(); rows=[]
    for n in [8,16,32]:
        Z=pool(X,n)
        for rep in range(repeats):
            rng=np.random.default_rng(20261002+rep); ug=np.unique(groups); order=rng.permutation(ug);rr={g:i for i,g in enumerate(order)};idx=np.argsort([rr[g] for g in groups])
            for fold,(tr0,te0) in enumerate(GroupKFold(folds).split(Z[idx],y[idx],groups[idx])):
                tr,te=idx[tr0],idx[te0];m=Pipeline([('scale',StandardScaler()),('model',ElasticNet(alpha=.01,l1_ratio=.5,max_iter=20000))]);m.fit(Z[tr],y[tr]);pred=np.expm1(m.predict(Z[te]));rm=np.sqrt(mean_squared_error(y0[te],pred));rows.append(dict(n_intervals=n,R2=r2_score(y0[te],pred),RMSE=rm,MAE=mean_absolute_error(y0[te],pred),repeat=rep,fold=fold))
    out=pd.DataFrame(rows);print(f'\n{fruit}/{camera}: interval mean pooling');print(out.groupby('n_intervals')[['R2','RMSE','MAE']].agg(['mean','std']).round(4).to_string());out.to_csv(ROOT/'baseline_cache'/f'intra_{fruit}_{camera}_interval_pool.csv',index=False)
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--fruit',default='Avocado');q.add_argument('--camera',default='VIS');q.add_argument('--repeats',type=int,default=5);q.add_argument('--folds',type=int,default=5);z=q.parse_args();main(z.fruit,z.camera,z.repeats,z.folds)
