"""Figures for the article. Reads the archived experiments without modifying them.

The standalone article embeds equivalent vector picture commands generated here;
its compilation does not require access to external image or data files.
"""
from pathlib import Path
from decimal import Decimal as D, getcontext
import csv,json,math,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures';SRC=ROOT/'figure_source';FROZEN=ROOT/'research_materials'
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
                     'axes.labelsize':9,'axes.titlesize':9.5,'legend.fontsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
styles=['-','--',':','-.'];markers=['o','s','^','D'];shades=['#172d3c','#4a4a4a','#6a6a6a','#838383']
sources={}
def readcsv(rel):
    p=FROZEN/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def readjson(rel):
    p=FROZEN/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def dumpcsv(name,rows):
    with (SRC/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def savefig(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

class Picture:
    def __init__(self,width=450,height=210):
        self.width=width;self.height=height
        self.rows=[r'\setlength{\unitlength}{1pt}',rf'\begin{{picture}}({width},{height})',r'\linethickness{0.55pt}']
    def text(self,x,y,s,align='c',size=r'\scriptsize'):
        self.rows.append(rf'\put({x:.3f},{y:.3f}){{\makebox(0,0)[{align}]{{{size} {s}}}}}')
    def line(self,x1,y1,x2,y2,width=.55):
        self.rows.append(rf'\linethickness{{{width:.2f}pt}}\qbezier({x1:.3f},{y1:.3f})({(x1+x2)/2:.3f},{(y1+y2)/2:.3f})({x2:.3f},{y2:.3f})')
    def box(self,x,y,w,h,lines):
        self.rows.append(rf'\put({x},{y}){{\framebox({w},{h}){{}}}}')
        for dy,text in lines:self.text(x+w/2,y+dy,text)
    def arrow(self,x,y,length):self.rows.append(rf'\put({x},{y}){{\vector(0,-1){{{length}}}}}')
    def mark(self,x,y,style):
        glyph=[r'\circ',r'\square',r'\triangle',r'\diamond'][style%4]
        self.text(x,y,'$'+glyph+'$',size=r'\footnotesize')
    def legend(self,items,y=7):
        space=self.width/len(items)
        for j,(label,style) in enumerate(items):
            x=j*space+8;self.line(x,y,x+18,y,.7);self.mark(x+9,y,style);self.text(x+24,y,label,'l')
    def finish(self,name):
        self.rows.append(r'\end{picture}')
        (SRC/(name+'.tex')).write_text('\n'.join(self.rows)+'\n',encoding='utf8')

class Axis:
    def __init__(self,p,x,y,w,h,xlim,ylim,xticks,yticks,xlabel,ylabel,title,logx=False,logy=False):
        self.p=p;self.x=x;self.y=y;self.w=w;self.h=h;self.xlim=xlim;self.ylim=ylim;self.logx=logx;self.logy=logy
        p.line(x,y,x+w,y);p.line(x,y,x,y+h)
        for v,label in xticks:
            xx=self.px(v);p.line(xx,y,xx,y-3);p.text(xx,y-10,label)
        for v,label in yticks:
            yy=self.py(v);p.line(x-3,yy,x,yy);p.text(x-7,yy,label,'r')
        p.text(x+w/2,y-25,xlabel);p.text(x+w/2,y+h+15,title)
        p.rows.append(rf'\put({x-38:.3f},{y+h/2:.3f}){{\makebox(0,0){{\rotatebox{{90}}{{\scriptsize {ylabel}}}}}}}')
    def px(self,v):
        fn=math.log if self.logx else lambda a:a
        lo,hi=map(fn,self.xlim);return self.x+self.w*(fn(float(v))-lo)/(hi-lo)
    def py(self,v):
        fn=math.log if self.logy else lambda a:a
        lo,hi=map(fn,self.ylim);return self.y+self.h*(fn(float(v))-lo)/(hi-lo)
    def curve(self,xx,yy,style=0,mark=True):
        points=[(self.px(x),self.py(y)) for x,y in zip(xx,yy)]
        # Line patterns and open symbols keep the vector plots readable in grayscale.
        for i,(s,t) in enumerate(zip(points[:-1],points[1:])):
            if len(points)>20:
                visible=[True,i%6<4,i%3==0,i%8<5][style%4]
                if visible:self.p.line(*s,*t,.65)
            else:
                subdivisions=12
                for j in range(subdivisions):
                    visible=[True,j%6<4,j%3==0,j%8<5][style%4]
                    if visible:
                        a=j/subdivisions;b=(j+1)/subdivisions
                        self.p.line(s[0]+a*(t[0]-s[0]),s[1]+a*(t[1]-s[1]),s[0]+b*(t[0]-s[0]),s[1]+b*(t[1]-s[1]),.65)
        if mark:
            step=max(1,(len(points)-1)//4)
            for i in range(0,len(points),step):self.p.mark(*points[i],style)
    def errorbar(self,x,low,high):
        xx=self.px(x);yl=self.py(low);yh=self.py(high)
        self.p.line(xx,yl,xx,yh,.4);self.p.line(xx-2,yl,xx+2,yl,.4);self.p.line(xx-2,yh,xx+2,yh,.4)

def theoretical_ratio(alpha,r,kind):
    def m(x):
        if kind=='f':return np.expm1(alpha*np.log1p(x))-x**alpha
        return -np.expm1(alpha*np.log1p(-x))+x**alpha
    return m(r*r)/m(r)

def make_mechanism():
    fig,ax=plt.subplots(figsize=(7.3,2.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,txt):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.008',edgecolor='black',facecolor='white',lw=.7))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',linespacing=1.6,fontsize=9)
    box(.08,.76,.84,.18,'Joint and marginal CF factorization at positive lags\n'+r'$Y=X+\varepsilon\quad\longrightarrow\quad D_\ell(u,v)$')
    boxes=[(.01,'Frequency scaling\n'+r'$Z_2/Z_1=a^\alpha$'+'\n'+r'$\alpha=\log(Z_2/Z_1)/\log a$'),(.35,'Doubled lag\n'+r'$Z_3/Z_1=q_{m,\alpha}(r)$'+'\n'+r'$r=q_{m,\alpha}^{-1}(Z_3/Z_1)$'),(.69,'Amplitude\n'+r'$Z_1=Cu^\alpha m_\alpha(r)$'+'\n'+r'$C=Z_1/[u^\alpha m_\alpha(r)]$')]
    for x,txt in boxes:box(x,.14,.30,.40,txt)
    for x in [.16,.50,.84]:ax.annotate('',xy=(x,.55),xytext=(x,.75),arrowprops={'arrowstyle':'->','lw':.7})
    ax.text(.5,.04,r'$Z_i=-dD_i>0,\quad r=e^{-\kappa h\ell},\quad \kappa=-\log r/(h\ell)$',ha='center',fontsize=9)
    savefig(fig,'figure_1_identification')
    p=Picture(height=178)
    p.box(12,123,426,44,[(31,r'Positive-lag joint and marginal CF factorization'),(13,r'$Y=X+\varepsilon\quad\longrightarrow\quad D_\ell(u,v)$')])
    for x in [76,225,374]:p.arrow(x,122,21)
    p.box(4,28,144,72,[(58,r'\textbf{Frequency scaling}'),(37,r'$Z_2/Z_1=a^\alpha$'),(17,r'$\alpha=\log(Z_2/Z_1)/\log a$')])
    p.box(153,28,144,72,[(58,r'\textbf{Doubled lag}'),(37,r'$Z_3/Z_1=q_{m,\alpha}(r)$'),(17,r'$r=q_{m,\alpha}^{-1}(Z_3/Z_1)$')])
    p.box(302,28,144,72,[(58,r'\textbf{Amplitude}'),(37,r'$Z_1=Cu^\alpha m_\alpha(r)$'),(17,r'$C=Z_1/[u^\alpha m_\alpha(r)]$')])
    p.text(225,10,r'$Z_i=-dD_i>0,\qquad r=e^{-\kappa h\ell},\qquad\kappa=-\log r/(h\ell)$')
    p.finish('figure_1_identification')

def make_ratios():
    aa=[1.15,1.5,1.8,1.95];rr=np.linspace(.002,.998,61);rows=[]
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.7),layout='constrained')
    p=Picture();ticks=[(x,f'${x:g}$') for x in [0,.25,.5,.75,1]]
    for k,kind in enumerate(['f','g']):
        axis=Axis(p,45+230*k,57,155,120,(0,1),(0,1),ticks,ticks,'$r$',rf'$q_{{{kind},\alpha}}(r)$',rf'\textbf{{({chr(97+k)})}} ${kind}$ shape')
        for j,alpha in enumerate(aa):
            yy=theoretical_ratio(alpha,rr,kind)
            rows.extend(dict(shape=kind,alpha=alpha,r=float(x),ratio=float(y)) for x,y in zip(rr,yy))
            axes[k].plot(rr,yy,styles[j],marker=markers[j],markevery=15,ms=3,mfc='white',color=shades[j],lw=1,label=rf'$\alpha={alpha:g}$')
            axis.curve(rr,yy,j)
        axes[k].set(xlim=(0,1),ylim=(0,1),xlabel='$r$',ylabel=rf'$q_{{{kind},\alpha}}(r)$',title=f'({chr(97+k)}) '+kind+' shape')
        axes[k].set_xticks([0,.25,.5,.75,1]);axes[k].grid(alpha=.12);axes[k].legend(frameon=False,loc='upper left')
    p.legend([(rf'$\alpha={a:g}$',j) for j,a in enumerate(aa)]);p.finish('figure_2_ratio_curves')
    dumpcsv('figure_2_ratio_values.csv',rows);savefig(fig,'figure_2_ratio_curves')

def make_frequency():
    data=readcsv('new_experiments/central_summary.csv')
    diag=readjson('verification/added_results_checks.json')['frequency_jacobian_diagnostics']
    freq=[.0625,.125,.25,.5]
    cases=[('clean','Clean'),('cauchy','Cauchy'),('strong_gaussian','Gaussian SD 2'),('uniform_zero','Uniform')]
    fig=plt.figure(figsize=(7.3,4.5),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[1.15,1])
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[:,1])]
    p=Picture(height=310);xt=[(x,t) for x,t in zip(freq,[r'$1/16$',r'$1/8$',r'$1/4$',r'$1/2$'])]
    native=[Axis(p,45,190,175,85,(freq[0],freq[-1]),(.5,1.04),xt,[(x,f'${x:g}$') for x in [.5,.7,.9,1]],'$u$','Batch inclusion',r'\textbf{(a)} Clean / Cauchy',logx=True),
            Axis(p,45,57,175,85,(freq[0],freq[-1]),(0,1.04),xt,[(x,f'${x:g}$') for x in [0,.5,1]],'$u$','Batch inclusion',r'\textbf{(b)} Severe errors',logx=True)]
    right=Axis(p,283,95,147,160,(freq[0],freq[-1]),(20,60),xt,[(x,f'${x}$') for x in [20,30,40,50,60]],'$u$',r'$\operatorname{cond}_2(J_\eta)$',r'\textbf{(c)} Population Jacobian',logx=True)
    rows=[]
    for j,(noise,label) in enumerate(cases):
        selected=[next(r for r in data if r['noise']==noise and int(r['n'])==32000 and r['rule']=='fixed_'+str(u)) for u in freq]
        yy=[float(r['batch_coverage']) for r in selected];lo=[float(r['batch_low']) for r in selected];hi=[float(r['batch_high']) for r in selected]
        ax=axes[j//2];axis=native[j//2]
        ax.errorbar(freq,yy,yerr=np.maximum(0,[np.array(yy)-lo,np.array(hi)-yy]),color=shades[j],ls=styles[j],marker=markers[j],mfc='white',mew=1,ms=4.5,capsize=2,lw=1,label=label)
        axis.curve(freq,yy,j)
        for u,l,h in zip(freq,lo,hi):axis.errorbar(u,l,h)
        for r in selected:rows.append({'noise':noise,'u':r['rule'].replace('fixed_',''),'N':r['N'],'success':r['success'],'batch_K':r['batch_K'],'batch_coverage':r['batch_coverage'],'batch_low':r['batch_low'],'batch_high':r['batch_high']})
    for i,ax in enumerate(axes[:2]):
        ax.axhline(.95,color='black',ls='--',lw=.6);axis=native[i];axis.p.line(axis.x,axis.py(.95),axis.x+axis.w,axis.py(.95),.3)
        ax.set(xscale='log',ylim=(.5,1.04) if i==0 else (0,1.04),xlabel='Fixed base frequency $u$',ylabel='All-attempt batch inclusion',title=['(a) Clean / Cauchy','(b) Severe errors'][i])
        ax.set_xticks(freq,['1/16','1/8','1/4','1/2']);ax.xaxis.set_minor_locator(NullLocator());ax.grid(alpha=.12)
    dd=[next(r for r in diag if r['u']==u) for u in freq];yy=[r['condition'] for r in dd]
    axes[2].plot(freq,yy,'-o',color='black',ms=4,mfc='white',lw=1);axes[2].set(xscale='log',ylim=(20,60),xlabel='Base frequency $u$',ylabel=r'$\mathrm{cond}_2(J_\eta)$',title='(c) Population Jacobian')
    axes[2].set_xticks(freq,['1/16','1/8','1/4','1/2']);axes[2].xaxis.set_minor_locator(NullLocator());axes[2].grid(alpha=.12);right.curve(freq,yy,0)
    handles0,labels0=axes[0].get_legend_handles_labels();handles1,labels1=axes[1].get_legend_handles_labels()
    fig.legend(handles0+handles1,labels0+labels1,loc='outside lower center',ncol=4,frameon=False,fontsize=8)
    p.legend([(label,j) for j,(_,label) in enumerate(cases)]);p.finish('figure_3_frequency')
    dumpcsv('figure_3_frequency_values.csv',rows);dumpcsv('figure_3_conditioning_values.csv',dd);savefig(fig,'figure_3_frequency')


def make_five():
    data=readcsv('new_experiments/five_summary.csv');conds=readcsv('new_experiments/five_conditioning.csv');aa=[1.5,1.8,1.9,1.95,1.98]
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.8),layout='constrained');p=Picture(height=214)
    xt=[(x,f'${x:g}$') for x in [1.5,1.7,1.9,1.98]]
    left=Axis(p,45,57,155,125,(1.5,1.98),(0,1.04),xt,[(x,f'${x:g}$') for x in [0,.25,.5,.75,1]],r'$\alpha$','Inversion success fraction',r'\textbf{(a)} Five-feature recovery')
    right=Axis(p,275,57,155,125,(1.5,1.98),(250,350000),xt,[(10**x,f'$10^{x}$') for x in [3,4,5]],r'$\alpha$',r'$\operatorname{cond}_2(J_F)$',r'\textbf{(b)} Fixed-coordinate Jacobian',logy=True)
    rows=[]
    for j,scale in enumerate([1.,2.]):
        ss=[next(r for r in data if float(r['alpha'])==a and float(r['noise_scale'])==scale) for a in aa]
        yy=[int(r['success'])/int(r['N']) for r in ss]
        axes[0].plot(aa,yy,styles[j],color=shades[j],marker=markers[j],mfc='white',ms=3,label=['MA','Twice MA'][j],lw=1);left.curve(aa,yy,j)
        rows.extend({'alpha':r['alpha'],'noise_scale':r['noise_scale'],'N':r['N'],'success':r['success'],'success_fraction':y} for r,y in zip(ss,yy))
    yy=[float(next(r for r in conds if float(r['alpha'])==a)['condition']) for a in aa]
    axes[1].semilogy(aa,yy,'-o',color='black',mfc='white',ms=3,lw=1);right.curve(aa,yy,0)
    axes[0].set(xlabel=r'$\alpha$',ylabel='Inversion success fraction',ylim=(0,1.04),title='(a) Five-feature recovery')
    axes[1].set(xlabel=r'$\alpha$',ylabel=r'$\mathrm{cond}_2(J_F)$',ylim=(250,350000),title='(b) Fixed-coordinate Jacobian')
    for ax in axes:ax.set_xticks([1.5,1.7,1.9,1.98]);ax.grid(alpha=.12)
    axes[0].legend(frameon=False);p.legend([('MA noise',0),('Twice MA noise',1)]);p.finish('figure_4_gaussian_endpoint')
    dumpcsv('figure_4_success_values.csv',rows);dumpcsv('figure_4_conditioning_values.csv',conds);savefig(fig,'figure_4_gaussian_endpoint')

def worked_example():
    getcontext().prec=80
    def power(x,a):return (a*x.ln()).exp()
    def g(a,r):return D(1)+power(r,a)-power(D(1)-r,a)
    k=D(1);alpha=D('1.8');C=D('.8');h=D('.05');ell=D(5);u=D('.5');a=D(2)
    r=(-k*h*ell).exp();z1=C*power(u,alpha)*g(alpha,r);z2=power(a,alpha)*z1;z3=C*power(u,alpha)*g(alpha,r*r)
    recovered_alpha=(z2/z1).ln()/a.ln();q=z3/z1;lo=D(0);hi=D(1)
    for _ in range(230):
        mid=(lo+hi)/2
        if g(recovered_alpha,mid*mid)/g(recovered_alpha,mid)<q:lo=mid
        else:hi=mid
    rr=(lo+hi)/2;kk=-rr.ln()/(h*ell);cc=z1/(power(u,recovered_alpha)*g(recovered_alpha,rr))
    result={key:str(v) for key,v in {'kappa':k,'alpha':alpha,'C':C,'h':h,'ell':ell,'u':u,'a':a,'r':r,'Z1':z1,'Z2':z2,'Z3':z3,'frequency_ratio':z2/z1,'lag_ratio':q,'recovered_alpha':recovered_alpha,'recovered_r':rr,'recovered_kappa':kk,'recovered_C':cc}.items()}
    result.update(direction=-1,scope='Population substitution in the existing formula; not a new Monte Carlo run',decimal_precision=80)
    (SRC/'worked_example.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    assert max(abs(kk-k),abs(cc-C),abs(recovered_alpha-alpha))<D('1e-60')

def main():
    OUT.mkdir(exist_ok=True);SRC.mkdir(exist_ok=True)
    make_mechanism();make_ratios();make_frequency();make_five();worked_example()
    manifest={'sources_sha256':sources,'theoretical_figures':['figure_1_identification','figure_2_ratio_curves'],
              'archived_data_figures':['figure_3_frequency','figure_4_gaussian_endpoint'],
              'new_simulations':0,'estimator_or_configuration_changes':0,
              'inline_vector_source':'The self-contained article embeds the generated .tex picture bodies.',
              'software':{'python':'3.13','numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (SRC/'figure_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('Four figures and standalone vector bodies generated from theory or archived results.')
if __name__=='__main__':main()
