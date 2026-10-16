from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys,json,time,hashlib
sys.dont_write_bytecode=True
import numpy as np
import research_core as c
ROOT=Path(__file__).parent
P=json.loads((ROOT/'simulation_protocol.json').read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def plans(n):return [dict(x,u=x['u']*(n/8000)**(-.15) if x['shrink'] else x['u']) for x in P['methods']]
def features(y,methods):
    cells=sorted(set(t for m in methods for t in c.cells(m['depth'],P['ell'],m['u'])))
    M=len(y)-2*P['ell'];ef={};result={}
    def exp(u):
        if u not in ef:ef[u]=np.exp(1j*u*y)
        return ef[u]
    for lag,u,v in cells:
        A=exp(u)[:M];B=exp(v)[lag:lag+M];mu=np.array([(A*B).mean(),A.mean(),B.mean()])
        result[(lag,u,v)]=(float(np.log(abs(mu[0]))-np.log(abs(mu[1]))-np.log(abs(mu[2]))),float(abs(mu).min()))
    return result
def record(job):
    ai,ki,ni,pid=job;a=P['alphas'][ai];k=P['kappas'][ki];n=P['n'][ni];seed=[P['seed'],ai,ki,ni,pid,0]
    x=c.prior.latent(seed,k,a,n)
    iid=P['iid_cauchy']*np.random.default_rng([*seed[:-1],201]).standard_cauchy(n+1)
    return a,k,n,seed,x,iid
def worker(job):
    a,k,n,seed,x,iid=record(job);methods=plans(n);rows=[]
    for idx,noise in enumerate(P['noises']):
        y=x+iid+c.noise_path([*seed[:-1],210+idx],n,noise,P['noise_scale']);F=features(y,methods)
        meta=dict(alpha=a,kappa=k,n=n,path_id=job[-1],noise=noise,latent_seed=seed,latent_sha256=c.prior.base.old.sha_array(x),observation_sha256=c.prior.base.old.sha_array(y))
        for m in methods:
            cc=c.cells(m['depth'],P['ell'],m['u']);ff=np.array([F[t][0] for t in cc]);B=c.eliminate(ff,m['depth']);estimate,status=c.inverse(B,m['depth'],P['ell'],m['u'])
            row=dict(meta,method=m['name'],u=m['u'],depth=m['depth'],status=status,B=B.tolist(),cf_min=min(F[t][1] for t in cc),loss=1.,accurate=False)
            if estimate is not None:
                kh,ah,Ch=map(float,estimate);error=(np.log(kh/k)/np.log(160))**2+((ah-a)/1.8)**2+(np.log(Ch/.8)/np.log(1000))**2
                row.update(kappa_hat=kh,alpha_hat=ah,C_hat=Ch,drift_relative_error=abs(kh/k-1),alpha_absolute_error=abs(ah-a),scale_relative_error=abs(Ch/.8-1),loss=min(1.,error/3),accurate=bool(abs(kh/k-1)<=.25 and abs(ah-a)<=.1 and abs(Ch/.8-1)<=.5))
            rows.append(row)
    return rows
def summarize(rows):
    keys=['alpha','kappa','n','noise','method'];groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
    out=[]
    for k,rr in groups.items():
        returned=[r for r in rr if r['status']=='returned']
        out.append(dict(zip(keys,k),paths=len(rr),return_rate=len(returned)/len(rr),accurate_rate=float(np.mean([r['accurate'] for r in rr])),all_path_mean_loss=float(np.mean([r['loss'] for r in rr])),conditional_median_drift_relative_error=float(np.median([r['drift_relative_error'] for r in returned])) if returned else None,conditional_median_alpha_error=float(np.median([r['alpha_absolute_error'] for r in returned])) if returned else None))
    return out
def main():
    frozen=json.loads((ROOT/'simulation_freeze.json').read_text(encoding='utf8'))
    for rel,digest in frozen['files'].items():assert sha(ROOT/rel)==digest,rel
    assert not (ROOT/'simulation_manifest.json').exists(),'Refuse replacement'
    start=time.time();manifest=dict(status='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),frozen_files=frozen['files'])
    (ROOT/'simulation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    jobs=[(ai,ki,ni,pid) for ai in range(len(P['alphas'])) for ki in range(len(P['kappas'])) for ni in range(len(P['n'])) for pid in range(P['path_id_start'],P['path_id_start']+P['paths_per_cell'])];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,rr in enumerate(pool.map(worker,jobs),1):
            rows.extend(rr)
            if i%240==0:print(json.dumps(dict(paths=i,total=len(jobs),seconds=time.time()-start)),flush=True)
    c.prior.base.writecsv(ROOT/'simulation_summary.csv',summarize(rows))
    c.prior.base.writecsv(ROOT/'simulation_pathwise.csv',rows)
    manifest.update(status='completed',latent_paths=len(jobs),observation_sets=len(jobs)*len(P['noises']),test_records=len(rows),seconds=time.time()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (ROOT/'simulation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
