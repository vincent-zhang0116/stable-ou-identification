from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,sys,time,hashlib
sys.dont_write_bytecode=True
import numpy as np
import smooth_noise_core as c
ROOT=Path(__file__).parent
P=json.loads((ROOT/'cusp_protocol.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def estimate(y,h):
    v=P['v'];ell=P['ell'];step=h/4;lo=-v*np.exp(-.05*.05*ell);hi=-v*np.exp(-8*.05*ell)
    centres=np.arange(lo,hi+step*.001,step);us=lo+np.arange(-8,len(centres)+8)*step
    M=len(y)-ell;first=y[:M];ev=np.exp(1j*v*y[ell:ell+M]);mb=ev.mean();vals=[];cfmin=1.
    for i in range(0,len(us),12):
        E=np.exp(1j*first[:,None]*us[None,i:i+12]);ma=E.mean(axis=0);mw=ev@E/M
        vals.extend(np.log(abs(mw))-np.log(abs(ma))-np.log(abs(mb)))
        cfmin=min(cfmin,float(abs(ma).min()),float(abs(mw).min()),float(abs(mb)))
    if cfmin<P['cf_gate']:return dict(status='cf_gate',cf_min=cfmin)
    z=np.asarray(vals);q=z[16:]-4*z[12:-4]+6*z[8:-8]-4*z[4:-12]+z[:-16]
    assert len(q)==len(centres)
    j=int(np.argmax(q));u=centres[j];kh=-np.log(-u/v)/(.05*ell)
    return dict(status='returned',kappa_hat=float(kh),cf_min=cfmin,boundary=bool(j==0 or j==len(centres)-1),contrast_peak=float(q[j]),grid_points=len(centres))
def worker(job):
    ai,pid=job;a=P['alphas'][ai];k=P['kappa'];n=P['n'];seed=[P['seed'],ai,pid,0]
    x=c.prior.latent(seed,k,a,n);iid=.1*np.random.default_rng([*seed[:-1],201]).standard_cauchy(n+1);rows=[]
    for ni,noise in enumerate(P['noises']):
        y=x+iid+c.noise_path([*seed[:-1],210+ni],n,noise)
        meta=dict(alpha=a,kappa=k,n=n,path_id=pid,noise=noise,latent_seed=seed,latent_sha256=c.prior.base.old.sha_array(x),observation_sha256=c.prior.base.old.sha_array(y))
        results=[]
        for h in P['bandwidths']:results.append((f'cusp_h{h}',estimate(y,h)))
        for name,depth,u in [('fixed5',1,.25),('low3',0,.125*(n/8000)**(-.15))]:
            F,cf,_,_=c.prior.feature_stats(y,c.cells(depth,5,u));B=c.eliminate(F,depth);th,status=c.inverse(B,depth,5,u)
            r=dict(status=status,cf_min=cf)
            if th is not None:r['kappa_hat']=float(th[0])
            results.append((name,r))
        for name,d in results:
            row=dict(meta,method=name,**d,accurate=False)
            if d['status']=='returned':row.update(drift_relative_error=abs(d['kappa_hat']/k-1),accurate=bool(abs(d['kappa_hat']/k-1)<=.25))
            rows.append(row)
    return rows
def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['alpha'],r['noise'],r['method']),[]).append(r)
    out=[]
    for (a,noise,method),rr in groups.items():
        returned=[r for r in rr if r['status']=='returned']
        out.append(dict(alpha=a,noise=noise,method=method,paths=len(rr),return_rate=len(returned)/len(rr),drift_accurate_rate=float(np.mean([r['accurate'] for r in rr])),conditional_median_relative_error=float(np.median([r['drift_relative_error'] for r in returned])) if returned else None,boundary_rate=float(np.mean([r.get('boundary',False) for r in rr]))))
    return out
def main():
    frozen=json.loads((ROOT/'cusp_freeze.json').read_text())
    for rel,h in frozen['files'].items():assert sha(ROOT/rel)==h
    assert not (ROOT/'cusp_manifest.json').exists()
    start=time.time();jobs=[(ai,pid) for ai in range(len(P['alphas'])) for pid in range(P['path_id_start'],P['path_id_start']+P['paths_per_cell'])];rows=[]
    (ROOT/'cusp_manifest.json').write_text(json.dumps(dict(status='running',frozen_files=frozen['files'])))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,rr in enumerate(pool.map(worker,jobs),1):
            rows.extend(rr)
            if i%40==0:print(json.dumps(dict(paths=i,seconds=time.time()-start)),flush=True)
    c.prior.base.writecsv(ROOT/'cusp_pathwise.csv',rows);c.prior.base.writecsv(ROOT/'cusp_summary.csv',summarize(rows))
    result=dict(status='completed',latent_paths=len(jobs),observations=len(jobs)*len(P['noises']),records=len(rows),seconds=time.time()-start)
    (ROOT/'cusp_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
