"""Reference implementations for the additional identification constructions.

These functions do not check independence, moment or mixing assumptions from
data. The endpoint output specifies matrix constraints, not a mesh confidence
set. Floating-point eigenvalues are diagnostics, not certified PSD arithmetic.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import toeplitz


def _record(y,lag):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not np.isfinite(y).all() or not 0<lag<len(y):
        raise ValueError('Need a finite one-dimensional record and a positive available lag')
    return y[:-lag],y[lag:]


def interaction_curve(y,lag,u,v,gate=.05,chunk=8):
    """Aligned log-modulus interactions at one lag and fixed second frequency."""
    first,last=_record(y,lag)
    u=np.atleast_1d(np.asarray(u,dtype=float))
    if not np.isfinite(u).all() or not np.isfinite(v) or not 0<gate<1:
        raise ValueError('Invalid frequencies or CF gate')
    ev=np.exp(1j*v*last);mb=ev.mean();out=np.empty(len(u));minimum=abs(mb)
    for start in range(0,len(u),chunk):
        E=np.exp(1j*first[:,None]*u[None,start:start+chunk])
        ma=E.mean(axis=0);joint=np.mean(E*ev[:,None],axis=0)
        minimum=min(minimum,float(abs(ma).min()),float(abs(joint).min()))
        if minimum<gate:return None,dict(status='cf_gate',cf_min=float(minimum))
        out[start:start+chunk]=np.log(abs(joint))-np.log(abs(ma))-np.log(abs(mb))
    return out,dict(status='available',cf_min=float(minimum),joint_frequencies=len(u))


def shape(alpha,x):
    return np.asarray(x)**alpha-np.expm1(alpha*np.log1p(-np.asarray(x)))


def low_frequency_inverse(y,h,ell,u,a=2.,b=.5,gate=.25,
                          kappa_box=(.05,8.),alpha_box=(.2,1.999),C_box=(.01,10.)):
    """Three interactions; finite-variance correlated noise causes finite-u bias."""
    if not(h>0 and ell>=1 and u>0 and a>1 and 0<b<=.5):
        raise ValueError('Invalid sampling or feature design')
    vals=[];mins=[]
    for lag,uu in [(ell,u),(ell,a*u),(2*ell,u)]:
        x,meta=interaction_curve(y,lag,[uu],-b*uu,gate)
        if x is None:return meta
        vals.append(x[0]);mins.append(meta['cf_min'])
    z=np.array(vals)
    if not np.isfinite(z).all() or np.any(z<=0):return dict(status='feature_sign',features=z.tolist())
    alpha=float(np.log(z[1]/z[0])/np.log(a))
    if not alpha_box[0]<=alpha<=alpha_box[1] or not 0<alpha<2:
        return dict(status='alpha_outside',features=z.tolist())
    q=z[2]/z[0]
    if not 0<q<1:return dict(status='lag_ratio',features=z.tolist())
    lo,hi=np.exp(-np.array(kappa_box[::-1])*h*ell)
    def f(r):return shape(alpha,b*r*r)/shape(alpha,b*r)-q
    if f(lo)>0 or f(hi)<0:return dict(status='drift_outside',features=z.tolist())
    r=brentq(f,lo,hi,xtol=2e-14)
    kappa=float(-np.log(r)/(h*ell));C=float(z[0]/(u**alpha*shape(alpha,b*r)))
    if not C_box[0]<=C<=C_box[1]:return dict(status='scale_outside',features=z.tolist())
    return dict(status='returned',kappa=kappa,alpha=alpha,C=C,features=z.tolist(),cf_min=min(mins))


def cusp_joint(y,h,ell,v,J,d,w,a=2.,gate=.05):
    """Theorem's deterministic grid and two distinct bandwidths.

    d localizes the cusp, w and a*w estimate its order and amplitude. A theorem
    concerns sequences with d/w -> 0, not a claim about any one finite choice.
    """
    if not(h>0 and v>0 and ell>=1 and -v<J[0]<J[1]<0 and d>0 and w>d and a>1):
        raise ValueError('Invalid search interval or bandwidths; require 0<d<w')
    extension=max(2*d,a*w)
    if J[1]+extension>=0:
        return dict(status='bandwidth_domain',reason='Stencil reaches the separate marginal singularity at zero')
    step=d/np.log(len(y)+1)
    grid=np.arange(J[0],J[1],step)
    grid=np.r_[grid,J[1]]
    all_u=np.vstack([grid-2*d,grid-d,grid,grid+d,grid+2*d,
                     grid-w,grid+w,grid-a*w,grid+a*w])
    unique,inverse=np.unique(all_u,return_inverse=True)
    values,meta=interaction_curve(y,ell,unique,v,gate)
    if values is None:return meta
    z=values[inverse].reshape(all_u.shape)
    contrast=z[4]-4*z[3]+6*z[2]-4*z[1]+z[0]
    at=int(np.argmax(contrast));uhat=grid[at]
    dw=z[5,at]+z[6,at]-2*z[2,at]
    daw=z[7,at]+z[8,at]-2*z[2,at]
    out=dict(kappa=float(-np.log(-uhat/v)/(h*ell)),u_hat=float(uhat),
             grid_points=len(grid),evaluated_u=len(unique),cf_min=meta['cf_min'],
             boundary=bool(at in [0,len(grid)-1]),delta_w=float(dw),delta_aw=float(daw))
    if dw>=0 or daw>=0:return dict(status='difference_sign',**out)
    alpha=float(np.log(daw/dw)/np.log(a))
    if not 0<alpha<2:return dict(status='alpha_outside',alpha=alpha,**out)
    C=float(-dw/(2*np.exp(alpha*np.log(w))))
    return dict(status='returned',alpha=alpha,C=C,**out)


def endpoint_constraints(y,m,u,v,t,beta_sum_bound,zeta=.05):
    """Return the exact-set formula's estimated coefficients and error radii.

    A known valid beta-sum bound is an input assumption, not an estimate.
    unrestricted=True means the whole prespecified parameter box must be kept.
    """
    if not(1<=m<len(y) and u!=0 and v!=0 and t!=0 and beta_sum_bound>=0 and 0<zeta<1):
        raise ValueError('Invalid lag count, frequencies, dependence bound or level')
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not np.isfinite(y).all():raise ValueError('Invalid record')
    N=len(y)-m;J=3*m+1
    variance=min(1.,(1+16*(m+beta_sum_bound))/N)
    delta=min(2.,np.sqrt(J*variance/zeta))
    lt=abs(np.mean(np.exp(1j*t*y)))-delta
    if lt<=0:return dict(unrestricted=True,reason='marginal_lower_modulus_zero',delta=delta)
    gamma=[];radii=[]
    for j in range(1,m+1):
        first,last=_record(y,j)
        A=np.exp(1j*u*first);D=np.exp(1j*v*last)
        mods=np.array([abs(np.mean(A*D)),abs(A.mean()),abs(D.mean())])
        lower=mods-delta
        if np.any(lower<=0):return dict(unrestricted=True,reason='interaction_lower_modulus_zero',delta=delta)
        gamma.append(float(-(np.log(mods[0])-np.log(mods[1])-np.log(mods[2]))/(u*v)))
        radii.append(float(delta*np.sum(1/lower)/abs(u*v)))
    return dict(unrestricted=False,lambda_upper=float(-2*np.log(lt)/t**2),
                gamma=gamma,radii=radii,inflation=2*sum(radii),delta=delta,
                m=m,zeta=zeta,beta_sum_bound=beta_sum_bound)


def endpoint_matrix(constraints,kappa,C,h):
    """Floating-point constraint matrix; membership is PSD in exact arithmetic."""
    if constraints['unrestricted']:return None
    m=constraints['m'];r=np.exp(-kappa*h)
    return (toeplitz(np.r_[constraints['lambda_upper'],constraints['gamma']])
            +constraints['inflation']*np.eye(m+1)-2*C*toeplitz(r**np.arange(m+1)))
