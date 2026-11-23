from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
out = parser.parse_args().output_dir
data = root / 'baseline_cache'
out.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})

abbr_fruit = {'Avocado':'Avo', 'Kaki':'Kak', 'Kiwi':'Kiw', 'Mango':'Man', 'Papaya':'Pap'}
pretty_dir = {'VIS->NIR':'VIS→NIR', 'NIR->VIS':'NIR→VIS', 'VIS->VIS_COR':'VIS→VC', 'VIS_COR->VIS':'VC→VIS'}
row_order = [('Avocado','NIR->VIS'),('Avocado','VIS->NIR'),('Avocado','VIS->VIS_COR'),('Avocado','VIS_COR->VIS'),('Kaki','VIS->VIS_COR'),('Kaki','VIS_COR->VIS'),('Kiwi','NIR->VIS'),('Kiwi','VIS->NIR'),('Mango','VIS->VIS_COR'),('Mango','VIS_COR->VIS'),('Papaya','VIS->VIS_COR'),('Papaya','VIS_COR->VIS')]
labels = [f'{abbr_fruit[f]}  {pretty_dir[d]}' for f,d in row_order]

budget = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted(data.glob('inter_calibration_budget_*.csv'))], ignore_index=True)
budget = budget[budget.method.eq('pair_ridge')]
assert budget['aggregate'].eq('repeat_pooled').all()
assert budget.groupby(['fruit','direction','budget_requested','repeat']).size().eq(5).all()
# Average subsample metrics within each repeat before averaging the five repeats.
budget = budget.groupby(['fruit','direction','budget_requested','repeat'], as_index=False).R2.mean()
budget = budget.groupby(['fruit','direction','budget_requested'], as_index=False).R2.mean()
budget.to_csv(out / 'Figure4A_plotted_data.csv', index=False, float_format='%.17g')
budgets = [4,8,16,24,32]
bm = np.full((len(row_order), len(budgets)), np.nan)
for i,(f,d) in enumerate(row_order):
    for j,b in enumerate(budgets):
        q = budget[(budget.fruit==f)&(budget.direction==d)&(budget.budget_requested==b)]
        if not q.empty: bm[i,j] = q.R2.iloc[0]

methods = ['raw','pair_ridge','storage_day_ridge']
sm = np.full((len(row_order), len(methods)), np.nan)
for i,(f,d) in enumerate(row_order):
    camera_pair = 'VIS_NIR' if 'NIR' in d else 'VIS_VIS_COR'
    storage = pd.read_csv(data / f'inter_storage_holdout_{f}_{camera_pair}.csv', float_precision='round_trip')
    for j,m in enumerate(methods):
        q=storage[(storage.fruit==f)&(storage.method==m)]
        if m!='storage_day_ridge': q=q[q.direction==d]
        if not q.empty: sm[i,j]=q.R2.iloc[0]
sm_color=np.sign(sm)*np.log10(1+np.abs(sm))
assert np.isfinite(bm).all() and np.isfinite(sm).all()
pd.DataFrame([dict(fruit=f, direction=d, method=m, R2=sm[i,j], color_value=sm_color[i,j])
              for i,(f,d) in enumerate(row_order) for j,m in enumerate(methods)]).to_csv(
                  out / 'Figure4B_plotted_data.csv', index=False, float_format='%.17g')

fig,(ax_a,ax_b)=plt.subplots(1,2,figsize=(8.35,6.15),gridspec_kw={'width_ratios':[1.32,1.11],'wspace':0.32})
# A
im_a=ax_a.imshow(bm,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.35,vcenter=0,vmax=.42),interpolation='nearest')
ax_a.set_title('A  Paired-calibration budget',loc='left',fontsize=12.5,fontweight='bold',pad=10)
ax_a.set_xticks(np.arange(len(budgets)),[str(x) for x in budgets]); ax_a.set_xlabel('Nominal calibration budget (pairs)',fontsize=11)
ax_a.set_yticks(np.arange(len(labels)),labels); ax_a.set_ylabel('Fruit–camera direction',fontsize=11)
ax_a.tick_params(axis='y',length=0,pad=3,labelsize=11); ax_a.tick_params(axis='x',length=2.5,pad=3,labelsize=11)
for i in range(bm.shape[0]):
    for j in range(bm.shape[1]):
        v=bm[i,j]
        if np.isfinite(v): ax_a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=11,color='white' if abs(v)>=.20 else '#222')
for y in [3.5,5.5,7.5,9.5]: ax_a.axhline(y,color='white',lw=1.0)
for s in ax_a.spines.values(): s.set_visible(False)
ca=fig.colorbar(im_a,ax=ax_a,fraction=.045,pad=.025); ca.set_label('Pooled $R^2$',fontsize=11,labelpad=5); ca.ax.tick_params(labelsize=11,length=2)
# B
im_b=ax_b.imshow(sm_color,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=float(np.nanmin(sm_color)),vcenter=0,vmax=float(np.nanmax(sm_color))),interpolation='nearest')
ax_b.set_title('B  Later-storage holdout',loc='left',fontsize=12.5,fontweight='bold',pad=10)
ax_b.set_xticks(np.arange(3),['Raw','Pair Ridge','Time-only']); ax_b.set_xlabel('Model',fontsize=11)
ax_b.set_yticks(np.arange(len(labels))); ax_b.set_yticklabels([]); ax_b.tick_params(axis='y',length=0); ax_b.tick_params(axis='x',length=2.5,pad=3,labelsize=10.5)
for i in range(sm.shape[0]):
    for j in range(sm.shape[1]):
        v=sm[i,j]
        if np.isfinite(v):
            rgb = im_b.cmap(im_b.norm(sm_color[i,j]))[:3]
            luminance = np.dot(rgb, [0.2126, 0.7152, 0.0722])
            ax_b.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=11,color='white' if luminance < .45 else '#222')
for y in [3.5,5.5,7.5,9.5]: ax_b.axhline(y,color='white',lw=1.0)
for s in ax_b.spines.values(): s.set_visible(False)
cb=fig.colorbar(im_b,ax=ax_b,fraction=.055,pad=.035); cb.set_label(r'Signed $\log_{10}(1+|R^2|)$',fontsize=11,labelpad=5); cb.ax.tick_params(labelsize=11,length=2)
fig.subplots_adjust(left=.18,right=.925,top=.90,bottom=.105)
stem='Figure4_calibration_and_storage_revised'
for ext,kwargs in [('pdf',{}),('svg',{}),('png',{'dpi':600})]: fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',**kwargs)
plt.close(fig)
