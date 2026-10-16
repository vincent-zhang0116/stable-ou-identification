"""v0.13 independently reconstructed stable-OU experiments.

All inferential functions take observations and a design, never simulation truth.
Truth is used only in the evaluation layer. Historical v0.10/v0.11 data are absent.
Run: OPENBLAS_NUM_THREADS=1 python source/experiment.py --config config.json --out results
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, platform, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy
from scipy.signal import lfilter
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import chi2, f, qmc

ROOT=Path(__file__).resolve().parents[1]

def sha_array(x):
    return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()

def stable(rng, alpha, size):
    """CMS, CF exp(-|t|**alpha), alpha != 1; no approximate Euler steps."""
    u=rng.uniform(-np.pi/2,np.pi/2,size); w=rng.exponential(size=size)
    return (np.sin(alpha*u)/np.cos(u)**(1/alpha)
            *(np.cos((1-alpha)*u)/w)**((1-alpha)/alpha))

def latent(seed, theta, h, n):
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    k,a,c=theta; ph=np.exp(-k*h)
    x0=c**(1/a)*stable(rng,a,1)[0]
    e=(c*(-np.expm1(-k*h*a)))**(1/a)*stable(rng,a,n)
    # x[0] already stationary. scipy's exact linear recursion in binary64.
    x=np.empty(n+1);x[0]=x0;x[1:]=lfilter([1.],[1.,-ph],e,zi=[ph*x0])[0]
    return x

def observed(x, name, seed):
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    if name=='clean':return x.copy()
    if name=='gaussian':return x+.4*rng.normal(size=len(x))
    if name=='strong_gaussian':return x+2*rng.normal(size=len(x))
    if name in ('cauchy','shifted_cauchy'):
        return x+.25*rng.standard_cauchy(size=len(x))+(2 if name=='shifted_cauchy' else 0)
    if name=='uniform_zero':return x+rng.uniform(-4*np.pi,4*np.pi,len(x))
    raise ValueError(name)

def cells(u,ell=5):return ((ell,u,-u),(ell,2*u,-2*u),(2*ell,u,-u))

def components(y,u):
    cs=cells(u);M=len(y)-10; U=[];mus=[]
    for ell,v,w in cs:
        A=np.exp(1j*v*y[:-ell]);B=np.exp(1j*w*y[ell:]);J=A*B
        block=np.column_stack([J,A,B]);mus.append(block.mean(0));U.append(block[:M])
    return np.concatenate(U,axis=1),np.asarray(mus)

def shape(a,r):return 1+r**a-(1-r)**a

def model(theta,h,u):
    th=np.atleast_2d(theta);k,a,c=th.T;r=np.exp(-5*h*k)
    z=c*u**a*shape(a,r)
    return np.column_stack([z,2**a*z,c*u**a*shape(a,r*r)])

def inversion(d,h,u,box):
    if not np.all(np.isfinite(d)) or np.any(d<=0):return 'feature_sign',None
    al=np.log(d[1]/d[0])/np.log(2);ratio=d[2]/d[0]
    if not 1<al<2:return 'alpha_outside_model',None
    if not 0<ratio<1:return 'lag_ratio_outside_model',None
    fun=lambda r:shape(al,r*r)/shape(al,r)-ratio
    lo=np.exp(-box[0][1]*h*5);hi=np.exp(-box[0][0]*h*5)
    if fun(lo)>0 or fun(hi)<0:return 'drift_outside_box',None
    r=brentq(fun,lo,hi,xtol=2e-14,rtol=2e-14)
    k=-np.log(r)/(5*h);c=d[0]/(u**al*shape(al,r));th=np.array([k,al,c])
    if any(not lo<=v<=hi for v,(lo,hi) in zip(th,box)):return 'parameter_outside_box',None
    return 'ok',th

def bartlett(z,B):
    z=z-z.mean(0);M=len(z)
    padded=np.pad(z,((B,B),(0,0)))
    cs=np.vstack([np.zeros(z.shape[1]),padded.cumsum(0)])
    win=cs[B+1:]-cs[:-B-1]
    return win.T@win/(M*(B+1))

def batch_cov(z,K):
    M=len(z);b=M//K
    means=z[:b*K].reshape(K,b,3).mean(1);means-=means.mean(0)
    return b*(means.T@means)/(K-1)

def infer(y,u,h,box,rho,K):
    U,mu=components(y,u);cm=float(np.abs(mu).min())
    ans={'mu':mu,'cf_min':cm,'point_status':'unsafe_cf','estimate':None,
         'd':None,'hac':None,'batch':None,'B':None}
    if not np.isfinite(mu).all() or cm<=rho:return ans
    lm=np.log(np.abs(mu));d=lm[:,0]-lm[:,1]-lm[:,2]
    ans['d']=d;ans['point_status'],ans['estimate']=inversion(d,h,u,box)
    q=(U/mu.reshape(1,9)).real.reshape(len(U),3,3)
    psi=q[:,:,0]-q[:,:,1]-q[:,:,2]+1
    B=min(len(U)-1,max(11,math.ceil(4*len(U)**.25)));ans['B']=B
    for name,g in [('hac',bartlett(psi,B)),('batch',batch_cov(psi,K))]:
        try:
            np.linalg.cholesky(g)
            ans[name]=g
        except np.linalg.LinAlgError:pass
    return ans

def choose_frequency(y,u0,tau,maxlevel=30):
    for j in range(maxlevel+1):
        u=u0*2.**(-j);_,mu=components(y,u)
        if np.abs(mu).min()>=tau:return j,u
    raise RuntimeError('numerical frequency termination cap; no silent fallback')

def mixing_K(theta,h):
    """Explicit valid Fourier-density upper bound, independent of C and noise."""
    k,a,c=np.atleast_2d(theta).T;ph=np.exp(-k*h)
    A=gamma(2/a)/(np.pi*a); B=(2*a+1)/np.pi
    density_deriv_L1_upper=4*np.sqrt(A*B)
    return density_deriv_L1_upper*(2*gamma(1-1/a)/np.pi)/(-np.expm1(-k*h*a))**(1/a)

def variance_bound(theta,h,N,span):
    k,a,c=np.atleast_2d(theta).T;ph=np.exp(-k*h);K=mixing_K(theta,h)
    j0=np.maximum(0,np.floor(np.log(8*K)/(k*h))).astype(int)
    tail=j0+8*K*np.exp(-k*h*(j0+1))/(-np.expm1(-k*h))
    return np.minimum(1.,(1+2*span+2*tail)/N)

def honest_accept(mu,theta,h,u,n,zeta):
    th=np.atleast_2d(theta);D=model(th,h,u);res=[]
    for i,ell in enumerate([5,5,10]):
        N=n+1-ell
        ej=np.minimum(2.,np.sqrt(9*variance_bound(th,h,N,ell)/zeta))
        em=np.minimum(2.,np.sqrt(9*variance_bound(th,h,N,0)/zeta))
        a=np.exp(-D[:,i]);jhat,ahat,bhat=mu[i]
        rr=np.abs(a*jhat-ahat*bhat)
        res.append(rr<=a*ej+2*em+1e-14)
    return np.logical_and.reduce(res)

def ellipsoid_accept(result,theta,h,u,n,kind,critical):
    g=result[kind]
    if g is None:return None
    delta=result['d']-model(theta,h,u)
    return np.einsum('ij,ij->i',delta,np.linalg.solve(g,delta.T).T)*n<=critical

def worker(job):
    cfg,gi,pathid=job; group=cfg['groups'][gi];theta=group['theta'];ns=group['n'];h=cfg['h']
    seed=[cfg['master_seed'],gi,pathid,0];x=latent(seed,theta,h,max(ns))
    rows=[];features=[];raw={};geo_n=cfg['geometry_paths']
    if pathid<geo_n:
        pts=qmc.Sobol(3,scramble=True,seed=cfg['geometry_seed']).random_base2(cfg['geometry_log2'])
        lo=np.array([np.log(cfg['box'][0][0]),cfg['box'][1][0],np.log(cfg['box'][2][0])]);hi=np.array([np.log(cfg['box'][0][1]),cfg['box'][1][1],np.log(cfg['box'][2][1])])
        eta=lo+(hi-lo)*pts;cloud=np.column_stack([np.exp(eta[:,0]),eta[:,1],np.exp(eta[:,2])])
    for ni,noise in enumerate(group['noises']):
        # A shifted-Cauchy observation shares exactly the unshifted noise draw.
        noiseindex=group['noises'].index('cauchy') if noise=='shifted_cauchy' else ni
        nseed=[cfg['master_seed'],gi,pathid,100+noiseindex]
        y=observed(x,noise,nseed)
        if pathid<cfg['raw_paths']:raw[noise]=y
        for n in ns:
            yy=y[:n+1];level,u=choose_frequency(yy,cfg['u0'],cfg['tau'])
            designs=[('fixed',0,cfg['u0']),('adaptive',level,u)]
            cache={}
            for design,j,uu in designs:
                if uu not in cache:cache[uu]=infer(yy,uu,h,cfg['box'],cfg['rho'],cfg['batches'])
                fit=cache[uu];crit=chi2.ppf(.95,3)
                critbatch=3*(cfg['batches']-1)/(cfg['batches']-3)*f.ppf(.95,3,cfg['batches']-3)
                row={'group':group['name'],'group_id':gi,'path_id':pathid,'noise':noise,'n':n,'design':design,'level':j,'u':uu,
                     'theta_k':theta[0],'theta_a':theta[1],'theta_c':theta[2],
                     'latent_seed':json.dumps(seed),'noise_seed':json.dumps(nseed),'latent_sha256':sha_array(x),
                     'observation_prefix_sha256':sha_array(yy),'cf_min':fit['cf_min'],'point_status':fit['point_status'],
                     'B':fit['B'],'estimate_k':None,'estimate_a':None,'estimate_c':None,
                     'hac_issued':int(fit['hac'] is not None),'batch_issued':int(fit['batch'] is not None),
                     'hac_covered':0,'batch_covered':0,'honest_covered':0,
                     'hac_volume':None,'batch_volume':None,'honest_volume':None}
                if fit['estimate'] is not None:
                    row.update(zip(['estimate_k','estimate_a','estimate_c'],map(float,fit['estimate'])))
                for kind,critical in [('hac',crit),('batch',critbatch)]:
                    acc=ellipsoid_accept(fit,theta,h,uu,n,kind,critical)
                    row[kind+'_covered']=int(acc[0]) if acc is not None else 0
                    if pathid<geo_n:
                        acc=ellipsoid_accept(fit,cloud,h,uu,n,kind,critical)
                        row[kind+'_volume']=float(acc.mean()) if acc is not None else None
                zeta=.05 if design=='fixed' else .05*2.**(-j-1)
                row['honest_covered']=int(honest_accept(fit['mu'],theta,h,uu,n,zeta)[0])
                if pathid<geo_n:row['honest_volume']=float(honest_accept(fit['mu'],cloud,h,uu,n,zeta).mean())
                frow={'key':[gi,pathid,noise,n,design],'cf_real':fit['mu'].real.tolist(),'cf_imag':fit['mu'].imag.tolist(),
                      'interaction':fit['d'].tolist() if fit['d'] is not None else None,
                      'hac_cov':fit['hac'].tolist() if fit['hac'] is not None else None,
                      'batch_cov':fit['batch'].tolist() if fit['batch'] is not None else None}
                features.append(frow);rows.append(row)
    if pathid<cfg['raw_paths']:raw['latent']=x
    return rows,features,raw

def summarize(rows):
    groups={}
    for row in rows:groups.setdefault((row['group'],row['noise'],int(row['n']),row['design']),[]).append(row)
    out=[]
    for key,rs in sorted(groups.items()):
        ok=[r for r in rs if r['point_status']=='ok'];N=len(rs)
        d=dict(zip(['group','noise','n','design'],key));d.update(N=N,point_success=len(ok),median_level=float(np.median([int(r['level']) for r in rs])))
        for p in ['k','a','c']:
            err=np.array([float(r['estimate_'+p])-float(r['theta_'+p]) for r in ok])
            d['rmse_'+p]=float(np.sqrt(np.mean(err*err))) if len(err) else None
            d['bias_'+p]=float(err.mean()) if len(err) else None
        d['alpha_failure_loss']=float(np.mean([(float(r['estimate_a'])-float(r['theta_a']))**2 if r['point_status']=='ok' else 1. for r in rs]))
        for method in ['hac','batch','honest']:
            K=sum(int(r[method+'_covered']) for r in rs);S=N if method=='honest' else sum(int(r[method+'_issued']) for r in rs)
            d[method+'_K']=K;d[method+'_S']=S;d[method+'_coverage']=K/N
            z=1.959963984540054;p=K/N;den=1+z*z/N;ctr=(p+z*z/(2*N))/den;hw=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/den
            d[method+'_wilson_low']=float(ctr-hw);d[method+'_wilson_high']=float(ctr+hw)
            vals=[float(r[method+'_volume']) for r in rs if r.get(method+'_volume') not in (None,'')]
            d[method+'_median_volume']=float(np.median(vals)) if vals else None;d[method+'_geometry_N']=len(vals)
        out.append(d)
    return out

def writecsv(path,rows):
    with Path(path).open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config.json');ap.add_argument('--out',default='results');ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'raw_selected').mkdir(exist_ok=True)
    started=time.time();rows=[];features=[];jobs=[(cfg,gi,j) for gi,g in enumerate(cfg['groups']) for j in range(g['paths'])]
    freeze={'config_sha256':hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(started)),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'independent_latent_paths':len(jobs),'workers':args.workers}
    (out/'run_manifest.json').write_text(json.dumps(freeze,indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count,(rr,ff,raw) in enumerate(pool.map(worker,jobs,chunksize=2),1):
            rows.extend(rr);features.extend(ff)
            if raw:
                gi=rr[0]['group_id'];pi=rr[0]['path_id'];np.savez_compressed(out/'raw_selected'/f'g{gi}_path{pi}.npz',**raw)
            if count%50==0:print(json.dumps({'paths_finished':count,'total':len(jobs),'elapsed_seconds':round(time.time()-started,1)}),flush=True)
    writecsv(out/'pathwise.csv',rows)
    with (out/'feature_outputs.jsonl').open('w') as fp:
        for r in features:fp.write(json.dumps(r,allow_nan=False)+'\n')
    summary=summarize(rows);writecsv(out/'summary.csv',summary);(out/'summary.json').write_text(json.dumps(summary,indent=2))
    freeze.update(finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=time.time()-started,pathwise_rows=len(rows),feature_rows=len(features),status='completed')
    (out/'run_manifest.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(freeze),flush=True)

if __name__=='__main__':main()
