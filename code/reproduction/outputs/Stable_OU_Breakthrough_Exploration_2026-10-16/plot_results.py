from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
if __name__=='__main__':
    kk=np.geomspace(.05,8,1000);r=np.exp(-kk*.05);r0=np.exp(-.05)
    A=(1+r)/(1-r);A0=(1+r0)/(1-r0);cap=.8*np.minimum(A/A0,A0/A)
    fig,ax=plt.subplots(figsize=(7.1,4.4),layout='constrained')
    ax.fill_between(kk,0,cap,color='#287a96',alpha=.18,label='Feasible drift-scale pairs')
    ax.plot(kk,cap,color='#287a96',lw=2,label='Sharp scale boundary')
    ax.axhline(.4,color='#a4512b',ls='--',label='Additional scale bound: C >= 0.4')
    ax.axhline(.01,color='#777777',ls=':',label='Original scale bound: C >= 0.01')
    ax.scatter([1],[.8],color='#222222',s=30,zorder=4,label='Generating signal')
    ax.set(xscale='log',xlim=(.05,8),ylim=(0,.88),xlabel='Drift kappa',ylabel='Stable OU scale C',title='Gaussian endpoint: population identified set')
    ax.set_xticks([.05,.1,.5,1,2,4,8],[.05,.1,.5,1,2,4,8])
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='upper left')
    fig.savefig(ROOT/'Endpoint_Identified_Set.png',dpi=220);fig.savefig(ROOT/'Endpoint_Identified_Set.svg')
    rows=list(csv.DictReader((ROOT/'cusp_confirmation_summary.csv').open()))
    methods=[('cusp_h0.04','Cusp, bandwidth 0.04','#287a96'),('cusp_h0.02','Cusp, bandwidth 0.02','#75a8b8'),('fixed5','Direct five-feature inverse','#b46334'),('low3','Low-frequency three-feature inverse','#777777')]
    fig,ax=plt.subplots(figsize=(8.1,4.4),layout='constrained')
    x=np.arange(3);width=.19
    for j,(method,label,color) in enumerate(methods):
        yy=[np.mean([float(v['drift_accurate_rate']) for v in rows if v['method']==method and float(v['alpha'])==a]) for a in [1,1.8,1.95]]
        ax.bar(x+(j-1.5)*width,yy,width,label=label,color=color)
    ax.set_xticks(x,['1.0','1.8','1.95'])
    ax.set(ylim=(0,1.06),xlabel='Stability index alpha',ylabel='Fraction with drift relative error <= 25%',title='New drift values: point-estimation confirmation')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.legend(frameon=False,fontsize=8,loc='upper right')
    fig.savefig(ROOT/'Cusp_Confirmation.png',dpi=220);fig.savefig(ROOT/'Cusp_Confirmation.svg')
