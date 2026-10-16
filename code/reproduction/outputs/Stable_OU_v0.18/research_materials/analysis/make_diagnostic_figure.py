from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'new_experiments/central_summary.json').read_text())
F=json.loads((ROOT/'new_experiments/five_summary.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
freq=[.0625,.125,.25,.5]
for noise,label,color in [('clean','Clean','#183b56'),('cauchy','Cauchy','#007f86'),('strong_gaussian','Gaussian SD 2','#b46b16')]:
    rows=[next(r for r in S if r['noise']==noise and r['n']==32000 and r['rule']=='fixed_'+str(u)) for u in freq]
    yy=np.array([r['batch_coverage'] for r in rows]);err=np.array([[r['batch_coverage']-r['batch_low'] for r in rows],[r['batch_high']-r['batch_coverage'] for r in rows]])
    ax[0].errorbar(freq,yy,yerr=np.maximum(0,err),label=label,color=color,marker='o',capsize=3,lw=1.3)
ax[0].axhline(.95,ls='--',color='#777777',lw=1);ax[0].set_xscale('log',base=2);ax[0].set_xticks(freq,[str(u) for u in freq]);ax[0].set_ylim(0,1.05)
ax[0].set(xlabel='Fixed base frequency u',ylabel='All-attempt batch coverage',title='A. CF safety does not guarantee calibration')
ax[0].legend(fontsize=8,frameon=False);ax[0].grid(alpha=.15)
aa=[1.5,1.8,1.9,1.95,1.98]
for scale,label,color in [(1.,'MA noise','#183b56'),(2.,'Twice MA noise','#b46b16')]:
    rows=[next(r for r in F if r['alpha']==a and r['noise_scale']==scale) for a in aa]
    ax[1].plot(aa,[r['success']/r['N'] for r in rows],label=label,marker='o',color=color)
ax[1].set(xlabel='Stable index alpha',ylabel='Five-feature inversion success',ylim=(0,1.05),title='B. Inversion deteriorates near alpha = 2')
ax[1].legend(fontsize=8,frameon=False);ax[1].grid(alpha=.15)
fig.savefig(ROOT/'supplement/diagnostics.png',dpi=220)
fig.savefig(ROOT/'supplement/diagnostics.svg')
