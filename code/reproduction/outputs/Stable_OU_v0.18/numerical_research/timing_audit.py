"""Registered single-worker timing replay; primary estimates remain unchanged."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import csv,json,statistics,time
import numpy as np
import benchmark as b

ROOT=Path(__file__).resolve().parent
def main():
    cfg=json.loads((ROOT/'design.json').read_text());cfg['horizons']=[8000]
    cfg['optimization_audit_first_paths']=0
    seed=cfg['confirmation_seed'];rows=[];checks=[];start=time.time()
    primary={}
    for module in ['central','gaussian']:
        with (ROOT/'confirmation'/f'{module}_pathwise.csv').open(encoding='utf8') as f:
            for r in csv.DictReader(f):
                key=(r['group'],int(r['path_id']),r['noise'],r['method']) if module=='central' else (float(r['alpha']),int(r['path_id']),float(r['noise_scale']),r['method'])
                primary[(module,key)]=r
    for rep in range(3):
        for module in ['central','gaussian']:
            jobs=[(cfg,seed,0,pid) for pid in range(10)] if module=='central' else [(cfg,seed,ai,pid) for ai in range(5) for pid in range(5)]
            for job in jobs:
                result=b.central_worker(job) if module=='central' else b.gaussian_worker(job)
                for r in result:
                    key=(r['group'],r['path_id'],r['noise'],r['method']) if module=='central' else (r['alpha'],r['path_id'],r['noise_scale'],r['method'])
                    p=primary[(module,key)];err=0.
                    for field in ['estimate_k','estimate_a','estimate_c','estimate_q5','estimate_q10']:
                        if p.get(field) not in [None,''] and r.get(field) is not None:err=max(err,abs(float(p[field])-float(r[field])))
                    checks.append({'module':module,'key':key,'status_match':p['status']==r['status'],'observation_match':p['observation_sha256']==r['observation_sha256'],'max_parameter_difference':err})
                    r.update(module=module,repetition=rep);rows.append(r)
        print('timing repetition',rep+1,'elapsed',round(time.time()-start,1),flush=True)
    b.writecsv(ROOT/'confirmation/timing_pathwise.csv',rows)
    summary=[]
    for module in ['central','gaussian']:
        for method in sorted({r['method'] for r in rows if r['module']==module}):
            rr=[r for r in rows if r['module']==module and r['method']==method]
            summary.append(dict(module=module,method=method,measurements=len(rr),median_method_seconds=statistics.median(r['total_method_seconds'] for r in rr),median_feature_seconds=statistics.median(r['feature_seconds'] for r in rr),median_estimation_seconds=statistics.median(r['estimation_seconds'] for r in rr),median_shared_selection_seconds=statistics.median(r['shared_selection_seconds'] for r in rr) if module=='central' else None,median_native_selection_seconds=statistics.median(r['native_selection_seconds'] for r in rr) if method=='three_native' else None))
    b.writecsv(ROOT/'confirmation/timing_summary.csv',summary)
    report={'scope':'registered subset, one worker, BLAS=1, three repetitions; does not replace primary estimates','elapsed_seconds':time.time()-start,'rows':len(rows),'all_statuses_match':all(c['status_match'] for c in checks),'all_observations_match':all(c['observation_match'] for c in checks),'maximum_parameter_difference':max(c['max_parameter_difference'] for c in checks),'summary':summary}
    (ROOT/'confirmation/timing_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    assert report['all_statuses_match'] and report['all_observations_match'] and report['maximum_parameter_difference']<1e-8
if __name__=='__main__':main()
