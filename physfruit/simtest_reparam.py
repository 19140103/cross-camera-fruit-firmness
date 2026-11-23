"""Separate PROSPECT physical scale from optimizer parameterization.

The reparameterized fit predicts log(c_i / N) rather than log(c_i). This
directly represents the per-layer absorption and avoids forcing Adam to move
large N and large concentrations together along a flat valley.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0,ROOT)
from physfruit.basis import camera_wavelengths  # noqa: E402
from physfruit.forward import prospect_RT  # noqa: E402

DT=torch.float64
ORDER=['cab','car','anth','brown','water','dm']
C_TRUE=np.array([35.,9.,2.,.05,.012,.007])

def inv_softplus(x):
    return np.log(np.expm1(max(float(x),1e-8)))

def fit(wl, N_true, noise, seed, reparam, steps=600):
    # The repository's PROSPECT basis is defined at the camera wavelengths;
    # this synthetic test uses the complementary VIS+NIR observation.
    from physfruit.simtest_range import basis_on
    vis=camera_wavelengths('VIS'); nir=camera_wavelengths('NIR')
    epsv,nrv=basis_on(vis); epsn,nrn=basis_on(nir)
    eps=torch.cat([epsv,epsn],dim=1); nr=torch.cat([nrv,nrn],dim=0)
    R0,_=prospect_RT(torch.tensor([[N_true]],dtype=DT),torch.tensor(C_TRUE[None],dtype=DT),eps,nr)
    g=torch.Generator().manual_seed(seed); y=R0+noise*torch.randn(R0.shape,generator=g,dtype=DT)
    gi=torch.Generator().manual_seed(1000+seed)
    raw_N=torch.tensor([[inv_softplus(N_true-1.)]],dtype=DT)+.4*torch.randn((1,1),generator=gi,dtype=DT); raw_N.requires_grad_()
    if reparam:
        raw_q=torch.tensor(np.log(C_TRUE/N_true)[None],dtype=DT)+.6*torch.randn((1,len(ORDER)),generator=gi,dtype=DT); raw_q.requires_grad_(); params=[raw_N,raw_q]
    else:
        raw_c=torch.tensor(np.log(C_TRUE)[None],dtype=DT)+.6*torch.randn((1,len(ORDER)),generator=gi,dtype=DT); raw_c.requires_grad_(); params=[raw_N,raw_c]
    opt=torch.optim.Adam(params,lr=.03)
    for _ in range(steps):
        opt.zero_grad(); N=1.+torch.nn.functional.softplus(raw_N)
        c=(N*torch.exp(raw_q)) if reparam else torch.exp(raw_c)
        R,_=prospect_RT(N,c,eps,nr); loss=((R-y)**2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        N=(1.+torch.nn.functional.softplus(raw_N)).item(); c=(N*torch.exp(raw_q) if reparam else torch.exp(raw_c)).squeeze().numpy();
        R,_=prospect_RT(torch.tensor([[N]],dtype=DT),torch.tensor(c[None],dtype=DT),eps,nr)
    return dict(N=N,c=c,mse=float(((R-y)**2).mean()),k=c/N)

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=600); ap.add_argument('--trials',type=int,default=3); a=ap.parse_args()
    print('N_true  parameterization  mse_med  N_med  N_spread%  k_spread%  c_spread%')
    rows=[]
    for nt in [6.,10.,20.,50.]:
        for rp in [False,True]:
            rs=[fit(None,nt,.002,s,rp,steps=a.steps) for s in range(a.trials)]
            def spr(v):
                v=np.asarray(v); return 100*(v.max()-v.min())/max(1e-12,abs(v.mean()))
            rows.append(dict(N_true=nt,param='k=c/N' if rp else 'c',mse=np.median([r['mse'] for r in rs]),N=np.median([r['N'] for r in rs]),Nspr=spr([r['N'] for r in rs]),kspr=spr([r['k'][0] for r in rs]),cspr=spr([r['c'][0] for r in rs])))
    for r in rows: print('%5.0f  %-16s %8.2e %7.3f %9.1f %9.1f %9.1f' % (r['N_true'],r['param'],r['mse'],r['N'],r['Nspr'],r['kspr'],r['cspr']))

if __name__=='__main__': main()
