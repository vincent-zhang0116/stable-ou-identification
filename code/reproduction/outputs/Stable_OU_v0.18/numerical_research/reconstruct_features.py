"""Post-run feature preservation from frozen seeds; no estimator is run or replaced.

Observation hashes and feature minima are checked against every primary row.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv,gzip,json,time
import numpy as np
import benchmark as b
R=Path(__file__).resolve().parent;cfg=json.loads((R/'design.json').read_text());seed=cfg['confirmation_seed']
def feature_record(y,row,kind,u):
    cells=b.grid(kind,u,cfg);vals,cm,mu=b.features(y,cells);assert b.old.sha_array(y)==row['observation_sha256'];assert abs(cm-float(row['cf_min']))<1e-12
    keys=['group','alpha','noise','noise_scale','path_id','n','method','observation_sha256','latent_sha256'];out={k:row[k] for k in keys if k in row}
    out.update(cells=cells,interaction=vals.tolist(),means_real=mu.real.tolist(),means_imag=mu.imag.tolist(),cf_min=cm,provenance='post-run reconstruction from frozen seeds; primary observation hash and minimum verified');return out
def worker(job):
    module,rr=job;row=rr[0];pid=int(row['path_id']);out=[]
    if module=='central':
        gi=next(i for i,g in enumerate(cfg['groups']) if g['name']==row['group']);x=b.old.latent([seed,gi,pid,0],cfg['groups'][gi]['theta'],cfg['h'],max(cfg['horizons']))
        for ni,noise in enumerate(cfg['noises']):
            y=b.old.observed(x,noise,[seed,gi,pid,100+ni])
            for r in [a for a in rr if a['noise']==noise]:
                kind={'three_shared':'three','three_native':'three','nine_md':'nine','thirtysix_md':'thirtysix'}[r['method']];u=float(r['native_u'] if r['method']=='three_native' else r['shared_u']);out.append(feature_record(y[:int(r['n'])+1],r,kind,u))
            if pid==0:np.savez_compressed(R/'confirmation/raw_selected'/f'iid_g{gi}_{noise}.npz',latent=x,observed=y)
    else:
        a=float(row['alpha']);ai=cfg['five_alphas'].index(a);n=cfg['five_n'];x=b.old.latent([seed,100+ai,pid,0],[1,a,.8],cfg['h'],n);e=cfg['five_iid_cauchy_scale']*np.random.default_rng([seed,100+ai,pid,201]).standard_cauchy(n+1);z=np.random.default_rng([seed,100+ai,pid,202]).normal(size=n+11);ma=.4*z[10:]+.4*z[5:-5]+.2*z[:-10]
        for r in rr:
            y=x+e+float(r['noise_scale'])*ma;out.append(feature_record(y,r,'five' if r['method']=='five_exact' else 'profile',cfg['five_base_frequency']))
        if pid==0:np.savez_compressed(R/'confirmation/raw_selected'/f'gaussian_a{ai}.npz',latent=x,iid_error=e,gaussian_MA=ma,observed_MA=x+e+ma,observed_twice_MA=x+e+2*ma)
    return module,out
def main():
    from collections import defaultdict
    jobs=[];start=time.time();dest=R/'confirmation';(dest/'raw_selected').mkdir(exist_ok=True)
    for module in ['central','gaussian']:
        g=defaultdict(list)
        with (dest/f'{module}_pathwise.csv').open(encoding='utf8') as f:
            for r in csv.DictReader(f):g[(r.get('group',r.get('alpha')),r['path_id'])].append(r)
        jobs.extend((module,rr) for rr in g.values())
    counts={'central':0,'gaussian':0}
    with gzip.open(dest/'reconstructed_features.jsonl.gz','wt',encoding='utf8') as f,ProcessPoolExecutor(max_workers=4) as pool:
        for i,(module,rows) in enumerate(pool.map(worker,jobs),1):
            for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
            counts[module]+=len(rows)
            if i%500==0:print(i,'paths features preserved;',round(time.time()-start,1),'sec',flush=True)
    assert counts=={'central':48000,'gaussian':4000}
    report={'counts':counts,'latent_paths':len(jobs),'all_primary_observation_hashes_match':True,'all_CF_minima_match_within_1e_minus_12':True,'provenance':'Reconstructed after confirmation from the unchanged frozen generator and primary row seeds/frequencies; not captured during the primary fitting run. No estimates replaced.','new_fits':0,'selected_raw_files':17,'elapsed_seconds':time.time()-start}
    (dest/'feature_reconstruction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
