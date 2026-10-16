"""Smooth-noise model exploration. No manuscript or earlier result is overwritten."""
from pathlib import Path
from decimal import Decimal as D,localcontext
import sys,os,json
sys.dont_write_bytecode=True
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent/'Stable_OU_Research_Validation_2026-10-16'))
# The original module has the same name; load it under a distinct module key.
import importlib.util
spec=importlib.util.spec_from_file_location('prior_validation_core',ROOT.parent/'Stable_OU_Research_Validation_2026-10-16/research_core.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

def coefficients(depth):
    p=np.array([1.])
    for j in range(1,depth+1):p=np.polynomial.polynomial.polymul(p,[-4.**j,1.])
    return p
def cells(depth,ell,u):
    return [(ell,2.**j*u,-.5*2.**j*u) for j in range(depth+2)]+[(2*ell,2.**j*u,-.5*2.**j*u) for j in range(depth+1)]
def eliminate(F,depth):
    p=coefficients(depth);return np.array([p@F[:depth+1],p@F[1:depth+2],p@F[depth+2:]])
def inverse(B,depth,ell,u):
    if not np.isfinite(B).all() or B[0]==0 or B[1]/B[0]<=0:return None,'frequency_sign'
    a=float(np.log(B[1]/B[0])/np.log(2.))
    if not .2<=a<2:return None,'alpha_outside'
    t=B[2]/B[0]
    if not 0<t<1:return None,'lag_ratio'
    lo=np.exp(-8*.05*ell);hi=np.exp(-.05*.05*ell)
    fun=lambda r:prior.g(a,.5*r*r)/prior.g(a,.5*r)-t
    if fun(lo)>0 or fun(hi)<0:return None,'drift_outside'
    r=brentq(fun,lo,hi,xtol=2e-14);k=-np.log(r)/(.05*ell)
    pp=np.polynomial.polynomial.polyval(2.**a,coefficients(depth))
    C=B[0]/(pp*u**a*prior.g(a,.5*r))
    if not .01<=C<=10:return None,'scale_outside'
    return np.array([k,a,C]),'returned'
def logcf(t,noise):
    if noise=='gaussian':return -.5*t*t
    if noise=='exponential':return -.5*np.log1p(t*t)
    if noise=='student3':return np.log1p(abs(t))-abs(t)
    if noise=='stable12':return -abs(t)**1.2
    raise ValueError(noise)
def ld(t,noise):
    t=D(t)
    if noise=='gaussian':return -t*t/2
    if noise=='exponential':return -(1+t*t).ln()/2
    if noise=='student3':return (1+abs(t)).ln()-abs(t)
    if noise=='stable12':return D(0) if t==0 else -(abs(t).ln()*D('1.2')).exp()
    raise ValueError(noise)
def pwr(x,a):return D(0) if x==0 else (abs(x).ln()*a).exp()
def population(k,a,lag,u,v,noise,scale=1):
    """70-digit log-CF interaction; iid errors cancel exactly."""
    with localcontext() as ctx:
        ctx.prec=70;k=D(str(k));a=D(str(a));u=D(str(u));v=D(str(v));scale=D(str(scale))
        r=(-k*D('.05')*lag).exp();signal=D('.8')*(pwr(u,a)+pwr(r*v,a)-pwr(u+r*v,a))
        left={-j:D(c)*scale for j,c in [(0,'.4'),(5,'.4'),(10,'.2')]}
        right={lag-j:D(c)*scale for j,c in [(0,'.4'),(5,'.4'),(10,'.2')]}
        total=D(0)
        for i in left.keys()|right.keys():
            x=u*left.get(i,D(0));y=v*right.get(i,D(0))
            total+=ld(x+y,noise)-ld(x,noise)-ld(y,noise)
        return +(signal+total)
def population_B(k,a,ell,u,noise,scale,depth):
    F=[population(k,a,l,x,y,noise,scale) for l,x,y in cells(depth,ell,u)]
    p=[D(str(x)) for x in coefficients(depth)]
    with localcontext() as ctx:
        ctx.prec=70
        return [sum(c*x for c,x in zip(p,F[:depth+1])),sum(c*x for c,x in zip(p,F[1:depth+2])),sum(c*x for c,x in zip(p,F[depth+2:]))]
def noise_path(seed,n,name,scale=1):
    rng=np.random.default_rng(seed)
    if name=='gaussian':z=rng.normal(size=n+11)
    elif name=='exponential':z=rng.exponential(size=n+11)-1
    elif name=='student3':z=rng.standard_t(3,size=n+11)/np.sqrt(3)
    elif name=='stable12':z=prior.cms(rng,1.2,0,n+11)
    else:raise ValueError(name)
    return scale*(.4*z[10:]+.4*z[5:-5]+.2*z[:-10])
