"""Numerical definitions for the frozen validation; no inference claim from code alone."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter
OUT=Path(__file__).parent
P=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
BASE=OUT.parent/'Stable_OU_v0.18/numerical_research'
spec=importlib.util.spec_from_file_location('frozen_benchmark',BASE/'benchmark.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CFG=json.loads((BASE/'design.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def g(alpha,x):return np.exp(alpha*np.log(x))-np.expm1(alpha*np.log1p(-x))
def phase_H(alpha,x):
    delta=alpha-1
    if abs(delta)<1e-12:return -2/np.pi*(x*np.log(x)+(1-x)*np.log1p(-x))
    return -(x*np.expm1(delta*np.log(x))+(1-x)*np.expm1(delta*np.log1p(-x)))/np.tan(np.pi*delta/2)
def cells3(ell,u):
    a=P['a'];b=P['b']
    return [(ell,u,-b*u),(ell,a*u,-b*a*u),(2*ell,u,-b*u)]
def cells5(ell,u):
    a=P['a'];b=P['b']
    return [(ell,u,-b*u),(ell,a*u,-b*a*u),(ell,a*a*u,-b*a*a*u),(2*ell,u,-b*u),(2*ell,a*u,-b*a*u)]
L=np.array([[-4.,1.,0.,0.,0.],[0.,-4.,1.,0.,0.],[0.,0.,0.,-4.,1.]])
def feature_stats(y,cells,influence=False):
    m=len(y)-max(c[0] for c in cells);cache={};ds=[];mus=[];psis=[]
    def expf(u):
        if u not in cache:cache[u]=cache[-u].conj() if -u in cache else np.exp(1j*u*y)
        return cache[u]
    for ell,u,v in cells:
        A=expf(u)[:m];B=expf(v)[ell:ell+m];W=A*B;mu=np.array([W.mean(),A.mean(),B.mean()]);mus.append(mu)
        ds.append(float(np.log(abs(mu[0]))-np.log(abs(mu[1]))-np.log(abs(mu[2]))))
        if influence:psis.append(np.real(W/mu[0]-A/mu[1]-B/mu[2])+1)
    mus=np.array(mus)
    return np.array(ds),float(abs(mus).min()),mus,np.array(psis).T if influence else None
def hac(psi,bandwidth):
    psi=psi-psi.mean(axis=0);m,d=psi.shape;B=min(int(bandwidth),m-1)
    padded=np.pad(psi,((B,B),(0,0)));cs=np.vstack([np.zeros((1,d)),np.cumsum(padded,axis=0)])
    wins=cs[B+1:]-cs[:-B-1]
    return (wins.T@wins)/(m*(B+1))
def model3(theta,ell,u):
    k,alpha,C=theta;r=np.exp(-k*P['h']*ell);z=C*u**alpha*g(alpha,P['b']*r)
    return np.array([z,P['a']**alpha*z,C*u**alpha*g(alpha,P['b']*r*r)])
def eliminated_unit(k,alpha,ell,u):
    r=np.exp(-k*P['h']*ell);t=P['a']**alpha;c=t-P['a']**2;z=u**alpha*g(alpha,P['b']*r);w=u**alpha*g(alpha,P['b']*r*r)
    return np.array([c*z,t*c*z,c*w])
def B_jacobian(theta,ell,u):
    eta=np.array([np.log(theta[0]),theta[1],np.log(theta[2])])
    def f(e):return np.exp(e[2])*eliminated_unit(np.exp(e[0]),e[1],ell,u)
    return np.column_stack([np.imag(f(eta.astype(complex)+1j*1e-25*np.eye(3)[j]))/1e-25 for j in range(3)])
def inverse3(d,ell,u,clip=False):
    if not np.isfinite(d).all() or np.any(d<=0):return None,'sign'
    alpha=np.log(d[1]/d[0])/np.log(P['a']);clipped=False
    if clip:
        aa=float(np.clip(alpha,P['box'][1][0],P['box'][1][1]));clipped=aa!=alpha;alpha=aa
    elif not P['box'][1][0]<=alpha<=P['box'][1][1]:return None,'alpha'
    rat=d[2]/d[0]
    if not 0<rat<1:return None,'ratio'
    lo=np.exp(-P['box'][0][1]*P['h']*ell);hi=np.exp(-P['box'][0][0]*P['h']*ell);b=P['b']
    fun=lambda r:g(alpha,b*r*r)/g(alpha,b*r)-rat
    if fun(lo)>0 or fun(hi)<0:return None,'kappa'
    r=brentq(fun,lo,hi,xtol=2e-14,rtol=2e-14);C=d[0]/(u**alpha*g(alpha,b*r));k=-np.log(r)/(P['h']*ell)
    if not P['box'][2][0]<=C<=P['box'][2][1]:return None,'scale'
    return np.array([k,alpha,C]),'alpha_clipped' if clipped else 'interior'
def fit3(d,ell,u,clip=False):
    th,st=inverse3(d,ell,u,clip=clip)
    if th is not None:return th,st,st=='alpha_clipped'
    th,st,meta=base.md_estimate(d,cells3(ell,u),P['h'],P['box'],CFG)
    return th,'md_'+st,bool(meta.get('boundary',False))
def cms(rng,alpha,skew,n):
    if skew==0:
        if alpha==1:return rng.standard_cauchy(n)
        if alpha==2:return np.sqrt(2)*rng.normal(size=n)
        return base.old.stable(rng,alpha,n)
    V=rng.uniform(-np.pi/2,np.pi/2,n);W=rng.exponential(size=n)
    if alpha==1:
        z=np.pi/2+skew*V
        return (2/np.pi)*(z*np.tan(V)-skew*np.log((np.pi/2)*W*np.cos(V)/z))
    tan=skew*np.tan(np.pi*alpha/2);B=np.arctan(tan)/alpha;S=(1+tan*tan)**(1/(2*alpha))
    return S*np.sin(alpha*(V+B))/np.cos(V)**(1/alpha)*(np.cos(V-alpha*(V+B))/W)**((1-alpha)/alpha)
def latent(seed,k,alpha,n,skew=0.):
    rng=np.random.default_rng(seed);ph=np.exp(-k*P['h']);C=P['C']
    if alpha==1:
        x0=C*cms(rng,alpha,skew,1)[0]+(2/np.pi)*C*skew*np.log(C)
        scale=C*(1-ph);loc=(2/np.pi)*C*skew*ph*np.log(ph)
        eps=scale*cms(rng,alpha,skew,n)+loc+(2/np.pi)*scale*skew*np.log(scale)
    else:
        x0=C**(1/alpha)*cms(rng,alpha,skew,1)[0];scale=(C*(-np.expm1(-k*P['h']*alpha)))**(1/alpha);eps=scale*cms(rng,alpha,skew,n)
    return np.r_[x0,lfilter([1.],[1.,-ph],eps,zi=[ph*x0])[0]]
def inference_record(case_index,n_index,pid):
    conf=P['inference'];n=conf['n'][n_index];case=conf['cases'][case_index]
    alpha=2-n**(-.25) if case=='local_n_quarter' else 2-n**(-.5) if case=='local_n_half' else float(case[6:])
    seed=[conf['seed'],case_index,n_index,pid,0];x=latent(seed,conf['kappa'],alpha,n)
    e=conf['iid_cauchy']*np.random.default_rng([conf['seed'],case_index,n_index,pid,201]).standard_cauchy(n+1)
    z=np.random.default_rng([conf['seed'],case_index,n_index,pid,202]).normal(size=n+11)
    ma=.4*z[10:]+.4*z[5:-5]+.2*z[:-10]
    return alpha,seed,x,e,ma
def adaptive_noise(ai,ki,pid,name,n):
    conf=P['adaptive'];ni=conf['noises'].index(name);rng=np.random.default_rng([conf['seed'],ai,ki,pid,100+ni])
    if name=='sparse_gaussian_ma10':
        z=rng.normal(size=n+11);return .4*z[10:]+.4*z[5:-5]+.2*z[:-10],{0,5,10},10
    if name=='dense_gaussian_ma3':
        z=rng.normal(size=n+4);return sum(c*z[3-j:n+4-j] for j,c in enumerate([.6,.5,.4,.3])),{0,1,2,3},3
    z=rng.standard_cauchy(n+4);return sum(c*z[3-j:n+4-j] for j,c in enumerate([.15,.1,.07,.04])),{0,1,2,3},3
def candidates(n):
    return [(s-j,j,.5*2.**(-j)) for s in range(1,int(np.ceil(np.log(n)))+1) for j in range(s)]
def skew_noise(ai,si,pid,name,n):
    conf=P['skew'];ni=conf['noises'].index(name);rng=np.random.default_rng([conf['seed'],ai,si,pid,100+ni])
    if name=='clean':return np.zeros(n+1)
    if name=='cauchy_0.25':return .25*rng.standard_cauchy(n+1)
    return .7*(rng.exponential(size=n+1)-1)
