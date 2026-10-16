from pathlib import Path
from decimal import Decimal as D,localcontext
import json,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.linalg import eigh,toeplitz
from scipy.optimize import brentq
import smooth_noise_core as c
ROOT=Path(__file__).parent
def capacity(k):
    r=np.exp(-k*.05);r0=np.exp(-.05);A=(1+r)/(1-r);A0=(1+r0)/(1-r0)
    return .8*min(A/A0,A0/A)
if __name__=='__main__':
    rows=[];checks=0
    omega=np.linspace(0,np.pi,20001)
    for k in [.05,.1,.5,1.,2.,4.,8.]:
        r=np.exp(-k*.05);r0=np.exp(-.05)
        s0=1.6*(1-r0*r0)/(1-2*r0*np.cos(omega)+r0*r0)
        sr=(1-r*r)/(1-2*r*np.cos(omega)+r*r)
        exact=capacity(k);grid=.5*float(np.min(s0/sr));assert abs(grid-exact)<1e-10
        previous=np.inf
        for N in [8,32,128]:
            T=toeplitz(1.6*r0**np.arange(N));R=toeplitz(r**np.arange(N))
            outer=.5*float(eigh(T,R,subset_by_index=[0,0],eigvals_only=True)[0])
            assert outer>=exact-1e-10 and outer<=previous+1e-10;previous=outer
            for fraction in [.5,1.]:
                assert np.linalg.eigvalsh(T-2*exact*fraction*R)[0]>-1e-9;checks+=1
            rows.append(dict(kappa=k,matrix_size=N,sharp_C_max=exact,finite_matrix_outer_C_max=outer))
    c.prior.base.writecsv(ROOT/'endpoint_capacity.csv',rows)
    A0=(1+np.exp(-.05))/(1-np.exp(-.05))
    kfrom=lambda A:-np.log((A-1)/(A+1))/.05
    intervals=dict(C_min_0p01_covers_original_drift_box=all(capacity(k)>=.01 for k in np.linspace(.05,8,1001)),C_min_0p4_drift_interval=[kfrom(2*A0),kfrom(A0/2)],C_known_0p8_point=1.)
    # Population fourth differences at the true cusp are oracle diagnostics.
    cusp=[]
    with localcontext() as ctx:
      ctx.prec=70
      for a in [.7,1.,1.8,1.95]:
       for noise in ['gaussian','exponential','student3']:
        k=1.;v=D('.5');u0=-(-D('.05')*5).exp()*v
        for d in [D('.04'),D('.01'),D('.001'),D('.0001')]:
            def delta(h):
                return c.population(k,a,5,u0+h,v,noise)-2*c.population(k,a,5,u0,v,noise)+c.population(k,a,5,u0-h,v,noise)
            z=delta(d);z2=delta(2*d);z4=delta(4*d)
            raw=float((z2/z).ln()/D(2).ln()) if z*z2>0 else None
            q=z2-4*z;q2=z4-4*z2
            quad=float((q2/q).ln()/D(2).ln()) if q*q2>0 else None
            cusp.append(dict(alpha=a,noise=noise,bandwidth=str(d),raw_alpha=raw,quadratic_filtered_alpha=quad,oracle_location=True))
    c.prior.base.writecsv(ROOT/'cusp_population.csv',cusp)
    # The nonnegative Fourier representation predicts unique peak at zero.
    maxima=[]
    xx=np.linspace(-10,10,20001)
    for a in [.2,.7,1.,1.8,1.95,1.999]:
        profile=-(abs(xx+2)**a-4*abs(xx+1)**a+6*abs(xx)**a-4*abs(xx-1)**a+abs(xx-2)**a)
        j=int(np.argmax(profile));assert abs(xx[j])<1e-10
        assert abs(profile[j]-2*(4-2**a))<1e-10;maxima.append(dict(alpha=a,peak_location=float(xx[j]),peak_height=float(profile[j])))
    out=dict(status='passed',spectral_and_matrix_checks=checks,endpoint_example=intervals,cusp_profiles=maxima,cusp_population_records=len(cusp),scope='Deterministic numerical checks of analytic examples; not a proof of statistical coverage or general sharpness.')
    (ROOT/'endpoint_cusp_manifest.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
