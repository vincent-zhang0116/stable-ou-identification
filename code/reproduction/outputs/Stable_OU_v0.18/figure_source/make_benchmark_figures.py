"""Generate paired confirmation figures; all-attempt denominators are retained."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib.pyplot as plt
from make_figures import Picture,Axis,savefig
R=Path(__file__).resolve().parents[1];S=R/'figure_source';D=R/'numerical_research/confirmation'
read=lambda p:list(csv.DictReader(p.open(encoding='utf8')))
def main():
    paired=read(D/'central_paired.csv');gauss=read(D/'gaussian_summary.csv');records=[]
    noises=['clean','cauchy','strong_gaussian','uniform_zero'];labels=['Clean','Cauchy','Gauss.','Uniform'];groups=['central','near_one','fast'];titles=['Central','Index 1.15','Drift 4']
    fig,axs=plt.subplots(1,3,figsize=(7.15,2.65),sharey=True);p=Picture(450,189)
    for gi,(g,title,ax) in enumerate(zip(groups,titles,axs)):
        aa=Axis(p,44+gi*148,46,110,112,(-.3,3.3),(-1,.2),list(enumerate(labels)),[(-1,'-1'),(-.5,'-.5'),(0,'0')],'Error law',r'Loss difference' if gi==0 else '',title)
        aa.curve([-.3,3.3],[0,0],2,False);ax.axhline(0,color='.7',lw=.6)
        for mi,m in enumerate(['nine_md','thirtysix_md']):
            yy=[];err=[];xx=np.arange(4)+(mi-.5)*.13
            for n in noises:
                row=next(x for x in paired if x['group']==g and x['noise']==n and x['n']=='8000' and x['comparison']==m+' minus three_shared' and x['metric']=='joint_loss');records.append(row);yy.append(float(row['difference']));err.append(1.959963984540054*float(row['MCSE']))
            aa.curve(xx,yy,mi)
            for x,y,e in zip(xx,yy,err):
                xp=aa.px(x);lo=aa.py(y-e);hi=aa.py(y+e);p.line(xp,lo,xp,hi);p.line(xp-2,lo,xp+2,lo);p.line(xp-2,hi,xp+2,hi)
            ax.errorbar(xx,yy,yerr=err,fmt=['o-','s--'][mi],mfc='white',ms=3,capsize=2,lw=.8,color=['#172d3c','#6a6a6a'][mi],label=['9 minus 3 shared','36 minus 3 shared'][mi])
        ax.set_xticks(range(4),labels,rotation=25);ax.set_ylim(-1,.2);ax.set_title(title);ax.set_xlabel('Error law')
    axs[0].set_ylabel('All-attempt loss difference');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.45),ncol=2,frameon=False);fig.subplots_adjust(bottom=.35,wspace=.2);savefig(fig,'figure_5_paired_benchmark');p.legend([('9 minus 3 shared',0),('36 minus 3 shared',1)],y=8);p.finish('figure_5_paired_benchmark')
    with (S/'figure_5_values.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    alphas=[1.5,1.8,1.9,1.95,1.98];fig,axs=plt.subplots(1,3,figsize=(7.15,2.6));p=Picture(450,190)
    for j,(metric,title) in enumerate([('return_fraction','Returned / attempted'),('boundary_fraction','Boundary / attempted'),('five_loss','Five-parameter loss')]):
        ax=axs[j];aa=Axis(p,44+148*j,49,110,110,(1.48,2.),(0,1.04),[(1.5,'1.5'),(1.8,'1.8'),(1.98,'1.98')],[(0,'0'),(.5,'.5'),(1,'1')],r'$\alpha$','Fraction / loss' if j==0 else '',title)
        for mi,(m,scale,label) in enumerate([('five_exact',1,'5, MA'),('five_exact',2,'5, 2 MA'),('twelve_profile',1,'12, MA'),('twelve_profile',2,'12, 2 MA')]):
            yy=[float(next(r for r in gauss if r['method']==m and float(r['noise_scale'])==scale and float(r['alpha'])==a)[metric]) for a in alphas]
            aa.curve(alphas,yy,mi);ax.plot(alphas,yy,['o-','s--','^:','D-.'][mi],color=['#172d3c','#4a4a4a','#6a6a6a','#838383'][mi],mfc='white',ms=3,lw=.8,label=label)
        ax.set_ylim(0,1.04);ax.set_xticks([1.5,1.8,1.98]);ax.set_xlabel(r'$\alpha$');ax.set_title(title)
    axs[0].set_ylabel('Fraction / bounded loss');axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.38),ncol=4,frameon=False);fig.subplots_adjust(bottom=.32,wspace=.35);savefig(fig,'figure_6_profile_endpoint');p.legend([('5, MA',0),('5, 2 MA',1),('12, MA',2),('12, 2 MA',3)],y=8);p.finish('figure_6_profile_endpoint')
    (S/'benchmark_figure_provenance.json').write_text(json.dumps({'source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['central_paired.csv','gaussian_summary.csv']},'independent_iid_paths_per_group':500,'gaussian_paths_per_alpha':200,'intervals':'pointwise descriptive mean +/- 1.96 paired MCSE; no simultaneous or inferential coverage claim','estimates_changed':0},indent=2))
if __name__=='__main__':main()
