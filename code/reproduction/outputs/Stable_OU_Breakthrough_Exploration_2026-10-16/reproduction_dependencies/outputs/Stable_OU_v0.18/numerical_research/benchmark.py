"""Isolated paired estimator research. Estimators receive observations/design only.

python numerical_research/benchmark.py --stage development --workers 4
python numerical_research/benchmark.py --stage confirmation --workers 4

Historical estimators/data are never overwritten. No confidence-coverage theorem
is claimed for computed multistart minima or the selected designs.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import argparse,csv,hashlib,json,math,os,platform,sys,time
import numpy as np
import scipy
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research_materials/original_v013/source'))
sys.path.insert(0,str(ROOT/'research_materials/analysis'))
import experiment as old
import new_experiments as legacy_diagnostic
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def grid(kind,u,cfg):
    if kind=='three':return [(5,u,-u),(5,2*u,-2*u),(10,u,-u)]
    if kind=='five':return [(5,u,-u),(5,2*u,-2*u),(5,4*u,-4*u),(10,u,-u),(10,2*u,-2*u)]
    if kind=='profile':return [(ell,m*u,-m*u) for ell in [5,10] for m in cfg['profile_frequency_multipliers']]
    ds=[-1.] if kind=='nine' else cfg['thirtysix_directions']
    return [(ell,m*u,d*m*u) for ell in cfg['nine_lags'] for m in cfg['nine_frequency_multipliers'] for d in ds]

def features(y,cells):
    """Aligned lag-specific means; conjugate frequencies reused algebraically."""
    cache={};values=[];means=[]
    def exponential(f):
        if f not in cache:
            if -f in cache:cache[f]=cache[-f].conj()
            else:cache[f]=np.exp(1j*f*y)
        return cache[f]
    for ell,u,v in cells:
        A=exponential(u)[:-ell];B=exponential(v)[ell:];mu=np.array([(A*B).mean(),A.mean(),B.mean()]);means.append(mu)
        values.append(float(np.log(abs(mu[0]))-np.log(abs(mu[1]))-np.log(abs(mu[2]))))
    means=np.array(means);return np.array(values),float(abs(means).min()),means

def choose(y,kind,base,cfg):
    for j in range(cfg['selection_cap']+1):
        u=base*2.**(-j);vals,cm,mu=features(y,grid(kind,u,cfg))
        if np.isfinite(mu).all() and cm>=cfg['shared_selection_threshold']:return j,u
    return None,None

def kernels(x,cells,h):
    """Unit-C interaction and independent analytic derivatives in log(k), alpha."""
    k=np.exp(x[0]);a=x[1];cs=np.asarray(cells);ell,u,v=cs.T;r=np.exp(-k*h*ell);rv=r*v;w=u+rv
    au=np.abs(u);ar=np.abs(rv);aw=np.abs(w)
    H=au**a+ar**a-aw**a
    logpow=lambda z:np.where(z>0,z**a*np.log(np.maximum(z,1e-300)),0.)
    Ha=logpow(au)+logpow(ar)-logpow(aw)
    Hk=a*(ar**(a-1)*np.sign(rv)-aw**(a-1)*np.sign(w))*v*(-k*h*ell*r)
    return H,np.column_stack([Hk,Ha])

def md_estimate(d,cells,h,box,cfg,initial=None,profile=False,audit=False):
    """Profile C (and Gaussian q), then deterministic grid plus local minimizers.

    This algorithm does NOT certify a global objective gap. Boundary solutions
    are returned and explicitly counted, not labelled interior exact inversions.
    """
    started=time.perf_counter();low=np.array([np.log(box[0][0]),box[1][0]]);high=np.array([np.log(box[0][1]),box[1][1]])
    if profile:
        cs=np.asarray(cells);V=np.column_stack([-cs[:,1]*cs[:,2]*(cs[:,0]==ell) for ell in [5,10]])
        Q=np.linalg.qr(V,mode='reduced')[0];project=lambda z:z-Q@(Q.T@z)
    else:V=None;project=lambda z:z
    target=project(d);scale=max(float(target@target),1e-12)
    def value(x,jac=True):
        H,J=kernels(x,cells,h);H=project(H);J=project(J);den=float(H@H)
        C=float(np.clip(H@target/max(den,1e-300),box[2][0],box[2][1]));res=C*H-target
        val=float(res@res)/scale;grad=2*C*(J.T@res)/scale
        return (val,grad) if jac else (val,C)
    ngrid=cfg['audit_grid_size_per_axis'] if audit else cfg['coarse_grid_size_per_axis'];nstarts=cfg['audit_local_starts'] if audit else cfg['local_starts']
    candidates=np.array(list(product(np.linspace(low[0],high[0],ngrid),np.linspace(low[1],high[1],ngrid))))
    scores=np.array([value(x,False)[0] for x in candidates]);starts=[candidates[i] for i in np.argsort(scores)[:nstarts]]
    if initial is not None and np.isfinite(initial[:3]).all():starts[-1]=np.clip([np.log(initial[0]),initial[1]],low,high)
    solutions=[];records=[]
    for x0 in starts:
        fit=minimize(value,x0,method='L-BFGS-B',jac=True,bounds=list(zip(low,high)),options={'ftol':cfg['ftol'],'gtol':cfg['gtol'],'maxiter':cfg['maxiter'],'maxls':30})
        records.append({'start':x0.tolist(),'converged':bool(fit.success),'criterion':float(fit.fun),'nfev':int(fit.nfev),'message':str(fit.message)})
        if fit.success and np.isfinite(fit.x).all() and np.isfinite(fit.fun):solutions.append(fit)
    meta={'optimizer_runs':len(starts),'optimizer_converged':len(solutions),'optimizer_nfev':sum(x['nfev'] for x in records),'optimizer_global_gap_certified':False,'optimizer_details':records}
    if not solutions:return None,'optimizer_no_converged_start',meta
    fit=min(solutions,key=lambda x:x.fun);_,C=value(fit.x,False);theta=np.array([np.exp(fit.x[0]),fit.x[1],C]);boundary=bool(np.any(np.minimum(fit.x-low,high-fit.x)<1e-6) or min(C-box[2][0],box[2][1]-C)<1e-6)
    if profile:
        H,_=kernels(fit.x,cells,h);q=np.linalg.lstsq(V,d-C*H,rcond=None)[0];theta=np.r_[theta,q]
    meta.update(criterion=float(fit.fun),boundary=boundary,optimizer_seconds=time.perf_counter()-started,local_criterion_spread=max(x.fun for x in solutions)-fit.fun)
    return theta,'boundary' if boundary else 'interior',meta

def error_metrics(estimate,truth,box):
    if estimate is None:return {'success':0,'joint_loss':1.,'five_loss':1. if len(truth)>3 else None,**{f'estimate_{n}':None for n in ['k','a','c','q5','q10']},**{f'error_{n}':None for n in ['k','a','c','q5','q10']}}
    eta=lambda t:np.array([np.log(t[0]),t[1],np.log(t[2])])
    span=eta(np.array(box)[:,1])-eta(np.array(box)[:,0]);loss=min(1.,float(np.sum(((eta(estimate)-eta(truth))/span)**2)))
    result={'success':1,'joint_loss':loss,'five_loss':min(1.,float(np.sum(((eta(estimate)-eta(truth))/span)**2))+float(np.sum((np.asarray(estimate[3:])-truth[3:])**2))) if len(truth)>3 else None}
    for i,n in enumerate(['k','a','c','q5','q10']):result['estimate_'+n]=float(estimate[i]) if i<len(estimate) else None;result['error_'+n]=float(estimate[i]-truth[i]) if i<len(estimate) else None
    return result

def central_worker(job):
    cfg,seed,gi,pid=job;group=cfg['groups'][gi];truth=np.array(group['theta']);h=cfg['h'];latent_seed=[seed,gi,pid,0];x=old.latent(latent_seed,truth,h,max(cfg['horizons']));rows=[]
    for ni,noise in enumerate(cfg['noises']):
        noise_seed=[seed,gi,pid,100+ni];y=old.observed(x,noise,noise_seed)
        for n in cfg['horizons']:
            yy=y[:n+1];t=time.perf_counter();level,u=choose(yy,'thirtysix',cfg['base_frequency'],cfg);selection_seconds=time.perf_counter()-t
            base=dict(group=group['name'],path_id=pid,n=n,noise=noise,latent_seed=latent_seed,noise_seed=noise_seed,latent_sha256=old.sha_array(x),observation_sha256=old.sha_array(yy),shared_level=level,shared_u=u,shared_selection_seconds=selection_seconds)
            if u is None:
                for name in ['three_shared','nine_md','thirtysix_md']:rows.append(dict(base,method=name,status='selection_cap',**error_metrics(None,truth,cfg['box'])))
            else:
                # All primary methods use precisely the same selected base frequency.
                cached={};cost={}
                for kind in ['three','nine','thirtysix']:
                    t=time.perf_counter();cached[kind]=features(yy,grid(kind,u,cfg));cost[kind]=time.perf_counter()-t
                avalues,acm,_=cached['three'];t=time.perf_counter();astatus,ath=old.inversion(avalues,h,u,cfg['box']) if acm>cfg['cf_floor'] else ('unsafe_cf',None);aseconds=time.perf_counter()-t
                ar=dict(base,method='three_shared',status=astatus,cf_min=acm,feature_seconds=cost['three'],estimation_seconds=aseconds,total_method_seconds=cost['three']+aseconds,boundary=False,**error_metrics(ath,truth,cfg['box']));rows.append(ar)
                # Alternating estimator order prevents fixed warm-up order favouring one MD method.
                order=['nine','thirtysix'] if pid%2==0 else ['thirtysix','nine']
                for kind in order:
                    vals,cm,mu=cached[kind];t=time.perf_counter();th,status,meta=md_estimate(vals,grid(kind,u,cfg),h,cfg['box'],cfg,initial=ath) if cm>cfg['cf_floor'] else (None,'unsafe_cf',{});elapsed=time.perf_counter()-t
                    row=dict(base,method=kind+'_md',status=status,cf_min=cm,feature_seconds=cost[kind],estimation_seconds=elapsed,total_method_seconds=cost[kind]+elapsed,**meta,**error_metrics(th,truth,cfg['box']))
                    if pid<cfg['optimization_audit_first_paths'] and n==max(cfg['horizons']) and noise in ['clean','strong_gaussian']:
                        auditth,auditstatus,ameta=md_estimate(vals,grid(kind,u,cfg),h,cfg['box'],cfg,initial=ath,audit=True)
                        row.update(audit_status=auditstatus,audit_criterion=ameta.get('criterion'),audit_criterion_improvement=None if th is None or auditth is None else meta['criterion']-ameta['criterion'],audit_parameter_distance=None if th is None or auditth is None else float(np.linalg.norm(th-auditth)))
                    rows.append(row)
            # Historical exact inverse with its own three-cell first-safe rule is an ablation.
            t=time.perf_counter();nativelevel,nativeu=choose(yy,'three',cfg['base_frequency'],cfg);nativeselect=time.perf_counter()-t
            if nativeu is None:
                rows.append(dict(base,method='three_native',status='selection_cap',native_u=None,native_level=None,**error_metrics(None,truth,cfg['box'])));continue
            t=time.perf_counter();vals,cm,mu=features(yy,grid('three',nativeu,cfg));nf=time.perf_counter()-t;t=time.perf_counter();status,th=old.inversion(vals,h,nativeu,cfg['box']) if cm>cfg['cf_floor'] else ('unsafe_cf',None);est=time.perf_counter()-t
            rows.append(dict(base,method='three_native',status=status,native_u=nativeu,native_level=nativelevel,native_selection_seconds=nativeselect,cf_min=cm,feature_seconds=nf,estimation_seconds=est,total_method_seconds=nf+est,boundary=False,**error_metrics(th,truth,cfg['box'])))
    return rows

def gaussian_worker(job):
    cfg,seed,ai,pid=job;alpha=cfg['five_alphas'][ai];n=cfg['five_n'];h=cfg['h'];u=cfg['five_base_frequency'];latent_seed=[seed,100+ai,pid,0];x=old.latent(latent_seed,[1,alpha,.8],h,n)
    e=cfg['five_iid_cauchy_scale']*np.random.default_rng([seed,100+ai,pid,201]).standard_cauchy(n+1);z=np.random.default_rng([seed,100+ai,pid,202]).normal(size=n+11);ma=.4*z[10:]+.4*z[5:-5]+.2*z[:-10];rows=[]
    for scale in cfg['five_noise_scales']:
        y=x+e+scale*ma;truth=[1,alpha,.8,.24*scale**2,.08*scale**2];base=dict(alpha=alpha,path_id=pid,noise_scale=scale,n=n,u=u,latent_seed=latent_seed,latent_sha256=old.sha_array(x),observation_sha256=old.sha_array(y));initial=None
        for kind,name in [('five','five_exact'),('profile','twelve_profile')]:
            cells=grid(kind,u,cfg);t=time.perf_counter();vals,cm,mu=features(y,cells);feature_seconds=time.perf_counter()-t;t=time.perf_counter()
            if cm<=cfg['cf_floor']:th,status,meta=None,'unsafe_cf',{}
            elif kind=='five':status,th=legacy_diagnostic.five_inverse(vals,h,u,cfg['box']);meta={};initial=th
            else:th,status,meta=md_estimate(vals,cells,h,cfg['box'],cfg,initial=initial,profile=True)
            elapsed=time.perf_counter()-t;rows.append(dict(base,method=name,status=status,cf_min=cm,feature_seconds=feature_seconds,estimation_seconds=elapsed,total_method_seconds=feature_seconds+elapsed,**meta,**error_metrics(th,truth,cfg['box'])))
    return rows

def writecsv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','confirmation'],required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--module',choices=['all','central','gaussian'],default='all');args=ap.parse_args();cfg=json.loads((Path(__file__).with_name('design.json')).read_text(encoding='utf8'))
    dest=Path(__file__).parent/args.stage;dest.mkdir(exist_ok=True);seed=cfg[args.stage+'_seed'];manifest={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':args.stage,'module':args.module,'source_sha256':sha(Path(__file__)),'design_sha256':sha(Path(__file__).with_name('design.json')),'historical_estimator_sha256':sha(Path(old.__file__)),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'workers':args.workers,'seed':seed,'status':'running'}
    freeze=Path(__file__).with_name('confirmation_freeze.json')
    if args.stage=='confirmation':
        frozen=json.loads(freeze.read_text(encoding='utf8'));assert frozen['source_sha256']==manifest['source_sha256'] and frozen['design_sha256']==manifest['design_sha256'],'Confirmation source/design differ from freeze.'
    mp=dest/(args.module+'_run_manifest.json');assert not mp.exists(),'Refuse to overwrite a stage run; use a new documented version.';mp.write_text(json.dumps(manifest,indent=2),encoding='utf8');start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.module in ['all','central']:
            jobs=[(cfg,seed,gi,pid) for gi in range(len(cfg['groups'])) for pid in range(cfg[args.stage+'_paths_per_group'])];rows=[]
            for i,result in enumerate(pool.map(central_worker,jobs),1):
                rows.extend(result)
                if i%25==0:print(json.dumps({'module':'central','paths':i,'total':len(jobs),'seconds':round(time.time()-start,1)}),flush=True)
            writecsv(dest/'central_pathwise.csv',rows);manifest.update(central_latent_paths=len(jobs),central_rows=len(rows));print('central complete',len(rows),flush=True)
        if args.module in ['all','gaussian']:
            jobs=[(cfg,seed,ai,pid) for ai in range(len(cfg['five_alphas'])) for pid in range(cfg['five_'+args.stage+'_paths_per_alpha'])];rows=[]
            for i,result in enumerate(pool.map(gaussian_worker,jobs),1):
                rows.extend(result)
                if i%100==0:print(json.dumps({'module':'gaussian','paths':i,'total':len(jobs),'seconds':round(time.time()-start,1)}),flush=True)
            writecsv(dest/'gaussian_pathwise.csv',rows);manifest.update(gaussian_latent_paths=len(jobs),gaussian_rows=len(rows))
    manifest.update(status='completed',elapsed_seconds=time.time()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));mp.write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
