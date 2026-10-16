"""Prespecified v0.15 diagnostics. No tuning on the archived v0.13 paths.
Run from the package root: python revision_source/new_experiments.py --workers 4
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, csv, hashlib, json, math, platform, sys, time
import numpy as np
import scipy
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'original_v013/source'))
import experiment as old


def five_model(eta, u, h):
    k, a, c = np.exp(eta[0]), eta[1], np.exp(eta[2])
    r = np.exp(-5*h*k)
    us = np.array([u,2*u,4*u,u,2*u])
    rs = np.array([r,r,r,r*r,r*r])
    qs = np.array([eta[3]]*3+[eta[4]]*2)
    return c*us**a*old.shape(a,rs)+qs*us**2


def five_inverse(values, h, u, box):
    b1,b2,b3 = values[1]-4*values[0],values[2]-4*values[1],values[4]-4*values[3]
    if not np.isfinite([b1,b2,b3]).all() or b1 == 0 or b2/b1 <= 0:
        return 'frequency_ratio', None
    a = np.log(b2/b1)/np.log(2)
    if not 1<a<2: return 'alpha_outside_model',None
    den = 2**a-4
    d = np.array([b1/den,2**a*b1/den,b3/den])
    status,th = old.inversion(d,h,u,box)
    if th is None: return status,None
    return 'ok',np.r_[th,(values[0]-d[0])/u**2,(values[3]-d[2])/u**2]


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;hw=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-hw,center+hw


def central(job):
    cfg,pid=job;theta=np.array(cfg['theta']);h=cfg['h'];nmax=max(cfg['horizons'])
    seed=[cfg['master_seed'],0,pid,0];x=old.latent(seed,theta,h,nmax)
    rows=[];details=[]
    for ni,noise in enumerate(cfg['noises']):
        y=old.observed(x,noise,[cfg['master_seed'],0,pid,100+ni])
        for n in cfg['horizons']:
            yy=y[:n+1];level,selected=old.choose_frequency(yy,.5,.2)
            cache={}
            rules=[('fixed_'+str(u),u,0) for u in cfg['frequencies']]+[('adaptive',selected,level)]
            for rule,u,j in rules:
                if u not in cache: cache[u]=old.infer(yy,u,h,cfg['box'],.05,20)
                fit=cache[u];ok=fit['estimate'] is not None
                row=dict(path_id=pid,n=n,noise=noise,rule=rule,u=u,level=j,point_status=fit['point_status'],
                         success=int(ok),cf_min=fit['cf_min'],latent_hash=old.sha_array(x),observation_hash=old.sha_array(yy))
                for pi,p in enumerate('kac'):
                    row['estimate_'+p]=float(fit['estimate'][pi]) if ok else None
                    row['error_'+p]=float(fit['estimate'][pi]-theta[pi]) if ok else None
                row['alpha_failure_loss']=float((fit['estimate'][1]-theta[1])**2) if ok else 1.
                for kind,critical in [('hac',old.chi2.ppf(.95,3)),('batch',3*19/17*old.f.ppf(.95,3,17))]:
                    acc=old.ellipsoid_accept(fit,theta,h,u,n,kind,critical)
                    row[kind+'_issued']=int(fit[kind] is not None)
                    row[kind+'_covered']=int(acc[0]) if acc is not None else 0
                    if fit[kind] is not None:
                        eig=np.linalg.eigvalsh(fit[kind]);row[kind+'_eig_ratio']=float(eig[0]/eig[-1])
                    else: row[kind+'_eig_ratio']=None
                rows.append(row)
                details.append(dict(key=[pid,noise,n,rule],cf_real=fit['mu'].real.tolist(),cf_imag=fit['mu'].imag.tolist(),
                                    interaction=None if fit['d'] is None else fit['d'].tolist(),
                                    hac=None if fit['hac'] is None else fit['hac'].tolist(),
                                    batch=None if fit['batch'] is None else fit['batch'].tolist()))
            # Scalar moment comparator: finite-moment recording errors only.
            if noise in ['clean','strong_gaussian']:
                instrument=np.sign(yy[:-10])*np.abs(yy[:-10])**.1
                m5=np.mean(yy[5:-5]*instrument);m10=np.mean(yy[10:]*instrument)
                ratio=m10/m5 if m5 != 0 else float('nan')
                k=-np.log(ratio)/(5*h) if 0<ratio<1 else None
                rows.append(dict(path_id=pid,n=n,noise=noise,rule='moment_drift',u=None,level=None,
                                 point_status='ok' if k is not None else 'moment_ratio',success=int(k is not None),
                                 estimate_k=k,error_k=None if k is None else float(k-theta[0]),
                                 latent_hash=old.sha_array(x),observation_hash=old.sha_array(yy)))
    return rows,details


def five(job):
    cfg,ai,pid=job;a=cfg['five_alphas'][ai];theta=[1.,a,.8];n=cfg['five_n'];h=cfg['h'];u=.25
    seed=[cfg['master_seed'],ai+1,pid,0];x=old.latent(seed,theta,h,n)
    rng=np.random.default_rng([cfg['master_seed'],ai+1,pid,201]);e=.1*rng.standard_cauchy(n+1)
    rng=np.random.default_rng([cfg['master_seed'],ai+1,pid,202]);z=rng.normal(size=n+11)
    ma=.4*z[10:]+.4*z[5:-5]+.2*z[:-10]
    rows=[]
    for scale in cfg['five_noise_scales']:
        y=x+e+scale*ma;vals=[];cfmin=1.
        for ell,freq in [(5,u),(5,2*u),(5,4*u),(10,u),(10,2*u)]:
            A=np.exp(1j*freq*y[:-ell]);B=np.exp(-1j*freq*y[ell:]);mu=np.array([(A*B).mean(),A.mean(),B.mean()])
            cfmin=min(cfmin,float(abs(mu).min()));vals.append(float(np.log(abs(mu[0]))-np.log(abs(mu[1]))-np.log(abs(mu[2]))))
        status,th=five_inverse(np.array(vals),h,u,cfg['box']) if cfmin>.05 else ('unsafe_cf',None)
        row=dict(alpha=a,path_id=pid,noise_scale=scale,cf_min=cfmin,point_status=status,success=int(th is not None),
                 latent_hash=old.sha_array(x),observation_hash=old.sha_array(y),feature_values=vals)
        target=[1.,a,.8,.24*scale**2,.08*scale**2]
        for i,p in enumerate(['k','a','c','q5','q10']):
            row['estimate_'+p]=None if th is None else float(th[i]);row['error_'+p]=None if th is None else float(th[i]-target[i])
        row['alpha_failure_loss']=float((th[1]-a)**2) if th is not None else 1.
        rows.append(row)
    return rows


def writecsv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=keys);w.writeheader();w.writerows(rows)


def summarize(rows,keys):
    groups={}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        d=dict(zip(keys,key));N=len(rs);S=sum(r['success'] for r in rs);d.update(N=N,success=S)
        for p in ['k','a','c','q5','q10']:
            es=[r['error_'+p] for r in rs if r.get('error_'+p) is not None]
            d['rmse_'+p]=float(np.sqrt(np.mean(np.square(es)))) if es else None
        for kind in ['hac','batch']:
            if kind+'_covered' not in rs[0]:continue
            K=sum(r[kind+'_covered'] for r in rs);I=sum(r[kind+'_issued'] for r in rs)
            lo,hi=wilson(K,N);d.update({kind+'_K':K,kind+'_issued':I,kind+'_coverage':K/N,kind+'_low':lo,kind+'_high':hi})
            vals=[r[kind+'_eig_ratio'] for r in rs if r[kind+'_eig_ratio'] is not None]
            d[kind+'_median_eig_ratio']=float(np.median(vals)) if vals else None
        vals=[r['alpha_failure_loss'] for r in rs if 'alpha_failure_loss' in r]
        d['alpha_failure_loss']=float(np.mean(vals)) if vals else None
        out.append(d)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    dest=ROOT/'new_experiments';cfg=json.loads((dest/'config.json').read_text());start=time.time()
    manifest=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  config_sha256=hashlib.sha256((dest/'config.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  design_status='fixed before execution; no retuning on outcomes')
    (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[];details=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(rr,ff) in enumerate(pool.map(central,[(cfg,p) for p in range(cfg['central_paths'])]),1):
            rows.extend(rr);details.extend(ff)
            if i%25==0:print('central',i,'seconds',round(time.time()-start),flush=True)
        five_rows=[]
        jobs=[(cfg,ai,p) for ai in range(len(cfg['five_alphas'])) for p in range(cfg['five_paths'])]
        for i,rr in enumerate(pool.map(five,jobs),1):
            five_rows.extend(rr)
            if i%100==0:print('five',i,'seconds',round(time.time()-start),flush=True)
    writecsv(dest/'central_pathwise.csv',rows)
    with (dest/'central_features.jsonl').open('w') as fp:
        for r in details:fp.write(json.dumps(r)+'\n')
    writecsv(dest/'five_pathwise.csv',five_rows)
    summaries=summarize(rows,['noise','n','rule']);fs=summarize(five_rows,['alpha','noise_scale'])
    writecsv(dest/'central_summary.csv',summaries);writecsv(dest/'five_summary.csv',fs)
    (dest/'central_summary.json').write_text(json.dumps(summaries,indent=2));(dest/'five_summary.json').write_text(json.dumps(fs,indent=2))
    # Paired uncertainty uses per-path differences, never independent-row standard errors.
    paired=[]
    for noise in cfg['noises']:
        for n in cfg['horizons']:
            by={(r['path_id'],r['rule']):r for r in rows if r['noise']==noise and r['n']==n}
            for rule in ['fixed_'+str(u) for u in cfg['frequencies']]:
                for metric in ['success','batch_covered','alpha_failure_loss']:
                    dif=np.array([by[p,rule][metric]-by[p,'adaptive'][metric] for p in range(cfg['central_paths'])])
                    paired.append(dict(noise=noise,n=n,comparison=rule+' minus adaptive',metric=metric,mean=float(dif.mean()),MCSE=float(dif.std(ddof=1)/np.sqrt(len(dif))),N=len(dif)))
    writecsv(dest/'paired_comparisons.csv',paired)
    conditioning=[]
    for a in cfg['five_alphas']:
        eta=np.array([0.,a,np.log(.8),.24,.08]);J=approx_derivative(lambda e:five_model(e,.25,cfg['h']),eta,method='3-point')
        sv=np.linalg.svd(J,compute_uv=False);conditioning.append(dict(alpha=a,smin=float(sv[-1]),condition=float(sv[0]/sv[-1])))
    writecsv(dest/'five_conditioning.csv',conditioning)
    manifest.update(status='completed',elapsed_seconds=time.time()-start,central_paths=cfg['central_paths'],five_paths=len(jobs),central_rows=len(rows),five_rows=len(five_rows))
    (dest/'run_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
