from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
out = parser.parse_args().output_dir
out.mkdir(parents=True, exist_ok=True)
data = root / 'baseline_cache'
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})

fruit = ['Avocado', 'Kaki', 'Kiwi', 'Mango', 'Papaya']
fruit_colors = {'Avocado':'#0072B2', 'Kaki':'#D55E00', 'Kiwi':'#009E73', 'Mango':'#E69F00', 'Papaya':'#CC79A7'}
directions = ['VIS→NIR', 'NIR→VIS', 'VIS→VC', 'VC→VIS']
markers = {'VIS→NIR':'o', 'NIR→VIS':'s', 'VIS→VC':'^', 'VC→VIS':'D'}

# One point per fruit–camera direction; x is the discrepancy reduction and y is Pair Ridge R^2.
main = pd.read_csv(data / 'main_results_frozen_2026-10-03.csv', float_precision='round_trip')
main = main.loc[main.method.eq('pair_ridge'), ['fruit', 'direction', 'R2_mean', 'R2_sd']]
mechanism = pd.concat([pd.read_csv(p, float_precision='round_trip')
                      for p in sorted(data.glob('inter_mechanism_diagnostics_*.csv'))])
assert mechanism['aggregate'].eq('repeat_pooled').all()
assert mechanism.groupby(['fruit', 'direction']).size().eq(5).all()
reduction = mechanism.groupby(['fruit', 'direction']).spectral_reduction_pct.mean().mul(100)
plotted = main.merge(reduction.rename('spectral_reduction_percent'),
                     on=['fruit', 'direction'], validate='one_to_one').sort_values(['fruit', 'direction'])
assert len(plotted) == 12 and np.isfinite(plotted.select_dtypes('number')).all().all()
plotted.to_csv(out / 'Figure3_plotted_data.csv', index=False, float_format='%.17g')
rows = [(r.fruit, r.direction.replace('->', '→').replace('VIS_COR', 'VC'),
         r.spectral_reduction_percent, r.R2_mean, r.R2_sd) for r in plotted.itertuples()]

fig, ax = plt.subplots(figsize=(7.0, 3.95))
ax.set_facecolor('#f4f5f6')
for f, d, x, y, yerr in rows:
    ax.errorbar(x, y, yerr=yerr, fmt=markers[d], ms=6.3,
                color=fruit_colors[f], ecolor=fruit_colors[f],
                elinewidth=0.9, capsize=2.5, capthick=0.9,
                markeredgecolor='white', markeredgewidth=0.45, zorder=3)
ax.axhline(0, color='#444444', lw=0.9, zorder=1)
ax.set_xlim(15, 92)
ax.set_ylim(-0.6, 0.5)
ax.set_xticks([20,30,40,50,60,70,80,90])
ax.set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4])
ax.set_xlabel('Reduction in standardized spectral discrepancy (%)')
ax.set_ylabel('Pair Ridge pooled $R^2$')
ax.grid(True, color='white', lw=0.8, zorder=0)
ax.tick_params(length=3)

fruit_handles = [Line2D([0],[0], marker='o', linestyle='None', color=fruit_colors[f],
                         markeredgecolor='white', markeredgewidth=.4, markersize=5.5, label=f) for f in fruit]
dir_handles = [Line2D([0],[0], marker=markers[d], linestyle='None', color='#444444',
                       markeredgecolor='white', markeredgewidth=.35, markersize=5.5, label=d) for d in directions]
# With five columns, interleaving the entries makes the first legend row the
# fruit colours and the second row the camera-direction markers.
legend_handles = []
legend_labels = []
for i, f_handle in enumerate(fruit_handles):
    legend_handles.append(f_handle)
    legend_labels.append(fruit[i])
    if i < len(dir_handles):
        legend_handles.append(dir_handles[i])
        legend_labels.append(directions[i])
ax.legend(legend_handles, legend_labels, ncol=5, loc='lower left',
          bbox_to_anchor=(0.018, 0.018), frameon=True, fontsize=8,
          facecolor='white', edgecolor='#c8cdd1', framealpha=0.94,
          borderpad=0.45, columnspacing=0.85, handletextpad=0.28,
          labelspacing=0.42, handlelength=1.0)
fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.18)
stem='Figure3_alignment_boundary_revised'
fig.savefig(out/f'{stem}.pdf', bbox_inches='tight')
fig.savefig(out/f'{stem}.svg', bbox_inches='tight')
fig.savefig(out/f'{stem}.png', dpi=600, bbox_inches='tight')
plt.close(fig)

