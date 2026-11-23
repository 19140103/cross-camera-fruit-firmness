from pathlib import Path
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'baseline_cache'
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
OUT = parser.parse_args().output_dir
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 8.5,
    'axes.labelsize': 10.0,
    'xtick.labelsize': 8.8,
    'ytick.labelsize': 8.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})

FABBR = {'Avocado': 'Avo', 'Kaki': 'Kak', 'Kiwi': 'Kiw', 'Mango': 'Man', 'Papaya': 'Pap'}
DABBR = {'VIS->NIR': 'VIS→NIR', 'NIR->VIS': 'NIR→VIS', 'VIS->VIS_COR': 'VIS→VC', 'VIS_COR->VIS': 'VC→VIS'}
ORDER = [('Avocado','NIR->VIS'),('Avocado','VIS->NIR'),('Avocado','VIS->VIS_COR'),('Avocado','VIS_COR->VIS'),('Kaki','VIS->VIS_COR'),('Kaki','VIS_COR->VIS'),('Kiwi','NIR->VIS'),('Kiwi','VIS->NIR'),('Mango','VIS->VIS_COR'),('Mango','VIS_COR->VIS'),('Papaya','VIS->VIS_COR'),('Papaya','VIS_COR->VIS')]
LABELS = [f'{FABBR[f]}  {DABBR[d]}' for f,d in ORDER]

plsr = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted(DATA.glob('inter_plsr_sensitivity_*.csv'))], ignore_index=True)
plsr = plsr[(plsr.source_model == 'plsr') & (plsr.transfer == 'pair_ridge')]
assert plsr['aggregate'].eq('repeat_pooled').all()
assert plsr.groupby(['fruit','direction','n_components']).size().eq(5).all()
plsr = plsr.groupby(['fruit','direction','n_components'], as_index=False).R2.mean()
plsr.to_csv(OUT / 'Figure5A_plotted_data.csv', index=False, float_format='%.17g')
components = [1,2,5,10,15]
panel_a = np.full((len(ORDER), len(components)), np.nan)
for i,(fruit,direction) in enumerate(ORDER):
    for j,n in enumerate(components):
        q=plsr[(plsr.fruit==fruit)&(plsr.direction==direction)&(plsr.n_components==n)]
        if not q.empty: panel_a[i,j]=q.R2.iloc[0]

null = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted(DATA.glob('inter_permutation_null_*.csv'))], ignore_index=True)
assert null['aggregate'].eq('repeat_pooled').all()
assert not null.duplicated(['fruit','direction','repeat','perm_id']).any()
assert null.groupby(['fruit','direction','repeat']).size().eq(50).all()
null['delta_pair_identity'] = null.R2_true_pair - null.R2
repeat_means = null.groupby(['fruit','direction','repeat'], as_index=False).delta_pair_identity.mean()
repeat_means.to_csv(OUT / 'Figure5B_repeat_means.csv', index=False, float_format='%.17g')
rows=[]
for (fruit,direction),g in null.groupby(['fruit','direction']):
    by_repeat=g.groupby('repeat').delta_pair_identity.mean()
    assert len(by_repeat) == 5 and len(g) == 250
    rows.append({'fruit':fruit,'direction':direction,'mean':by_repeat.mean(),
                 'sd_repeats':by_repeat.std(ddof=1),
                 'descriptive_q025':g.delta_pair_identity.quantile(.025),
                 'descriptive_q975':g.delta_pair_identity.quantile(.975),
                 'n_repeats':5,'n_draws':250})
panel_b=pd.DataFrame(rows).set_index(['fruit','direction']).reindex(ORDER).reset_index()
panel_b.to_csv(OUT / 'Figure5B_plotted_data.csv', index=False, float_format='%.17g')
assert np.isfinite(panel_a).all() and np.isfinite(panel_b.select_dtypes('number')).all().all()


def draw_pair_identity(ax):
    for i,row in enumerate(panel_b.itertuples(index=False)):
        ax.plot([row.descriptive_q025,row.descriptive_q975],[i,i],color='#B0B7BE',
                lw=3.5,solid_capstyle='butt',zorder=2)
        q=repeat_means[(repeat_means.fruit==row.fruit)&(repeat_means.direction==row.direction)].sort_values('repeat')
        ax.scatter(q.delta_pair_identity,i+np.linspace(-.17,.17,5),s=17,
                   facecolor='#0072B2',edgecolor='white',linewidth=.4,zorder=3)
        ax.plot(row.mean,i,marker='D',color='#202020',markeredgecolor='white',
                markeredgewidth=.5,markersize=5.2,zorder=4)
    ax.axvline(0,color='#555555',lw=.8,zorder=1)
    ax.set_xlim(-.32,.85)
    ax.set_ylim(11.6,-.6)
    ax.set_xticks([-.2,0,.2,.4,.6,.8])
    ax.set_xticklabels(['−0.2','0','0.2','0.4','0.6','0.8'])
    ax.set_yticks(np.arange(len(LABELS)),LABELS)
    ax.set_xlabel(r'$\Delta R^2$ (true pairing − shuffled pairing)',fontsize=9.5)
    ax.set_ylabel('Fruit–camera direction',fontsize=9.5)
    ax.set_title('B  Contribution of paired identity',loc='left',fontsize=11,fontweight='bold',pad=10)
    ax.tick_params(axis='x',labelsize=8.5,length=3)
    ax.tick_params(axis='y',labelsize=8.5,length=0,pad=4)
    ax.grid(axis='x',color='#E5E5E5',linewidth=.5,zorder=0)
    for yy in [3.5,5.5,7.5,9.5]:
        ax.axhline(yy,color='#E6E8EB',lw=.6,zorder=0)


legend_handles=[
    Line2D([0],[0],color='#B0B7BE',lw=3.5,label='2.5–97.5% range'),
    Line2D([0],[0],marker='o',linestyle='none',markerfacecolor='#0072B2',markeredgecolor='white',markersize=4.5,label='Repeat mean'),
    Line2D([0],[0],marker='D',linestyle='none',markerfacecolor='#202020',markeredgecolor='white',markersize=5,label='Overall mean')]


def export(fig, stem):
    for ext,kwargs in [('pdf',{}),('svg',{}),('png',{'dpi':600})]:
        fig.savefig(OUT/f'{stem}.{ext}',**kwargs)
    plt.close(fig)

fig,(ax_a,ax_b)=plt.subplots(2,1,figsize=(170/25.4,205/25.4),gridspec_kw={'height_ratios':[1,1.06],'hspace':.38})
# Panel A
a_norm=TwoSlopeNorm(vmin=-.40,vcenter=0,vmax=.45)
im=ax_a.imshow(panel_a,aspect='auto',cmap='RdBu_r',norm=a_norm,interpolation='nearest')
ax_a.set_title('A  PLSR source-model sensitivity',loc='left',fontsize=11,fontweight='bold',pad=10)
ax_a.set_xticks(np.arange(len(components)),[str(x) for x in components]); ax_a.set_xlabel('PLSR latent components',fontsize=9.5)
ax_a.set_yticks(np.arange(len(LABELS)),LABELS); ax_a.set_ylabel('Fruit–camera direction',fontsize=9.5)
ax_a.tick_params(axis='x',labelsize=8.5,length=3); ax_a.tick_params(axis='y',labelsize=8.5,length=0,pad=4)
for i in range(panel_a.shape[0]):
    for j in range(panel_a.shape[1]):
        v=panel_a[i,j]
        if np.isfinite(v): ax_a.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8.5,color='white' if abs(v)>=.22 else '#202020')
for y in [3.5,5.5,7.5,9.5]: ax_a.axhline(y,color='white',lw=1.25)
for spine in ax_a.spines.values(): spine.set_visible(False)
ca=fig.colorbar(im,ax=ax_a,fraction=.025,pad=.018); ca.set_label('Pair Ridge pooled $R^2$',fontsize=8.5); ca.ax.tick_params(labelsize=8,length=2)
# Panel B
draw_pair_identity(ax_b)
fig.subplots_adjust(left=.22,right=.90,top=.95,bottom=.11)
fig.legend(handles=legend_handles,frameon=False,ncol=3,loc='lower center',
           bbox_to_anchor=(.55,.018),handletextpad=.5,columnspacing=1.1,fontsize=8)
export(fig,'Figure5_source_sensitivity_pair_identity')

fig_b,ax=plt.subplots(figsize=(170/25.4,108/25.4))
draw_pair_identity(ax)
fig_b.subplots_adjust(left=.22,right=.96,top=.90,bottom=.20)
fig_b.legend(handles=legend_handles,frameon=False,ncol=3,loc='lower center',
             bbox_to_anchor=(.55,.012),handletextpad=.5,columnspacing=1.1,fontsize=8)
export(fig_b,'Figure5B_pair_identity')
