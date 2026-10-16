from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,time,sys,hashlib
sys.dont_write_bytecode=True
import cusp_pilot as cp
ROOT=Path(__file__).parent
P=json.loads((ROOT/'cusp_confirmation_protocol.json').read_text())
def worker(job):
    ai,ki,pid=job
    cp.P=dict(P,kappa=P['kappas'][ki])
    return cp.worker((ai,pid))
def summarize(rows):
    result=[]
    for k in P['kappas']:
        result.extend(dict(kappa=k,**r) for r in cp.summarize([r for r in rows if r['kappa']==k]))
    return result
if __name__=='__main__':
    freeze=json.loads((ROOT/'cusp_confirmation_freeze.json').read_text())
    for rel,digest in freeze['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
    assert not (ROOT/'cusp_confirmation_manifest.json').exists()
    start=time.time();rows=[];jobs=[(ai,ki,pid) for ai in range(len(P['alphas'])) for ki in range(len(P['kappas'])) for pid in range(P['path_id_starts'][ki],P['path_id_starts'][ki]+P['paths_per_cell'])]
    (ROOT/'cusp_confirmation_manifest.json').write_text(json.dumps(dict(status='running',frozen_files=freeze['files'])))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,rr in enumerate(pool.map(worker,jobs),1):
            rows.extend(rr)
            if i%90==0:print(json.dumps(dict(paths=i,seconds=time.time()-start)),flush=True)
    cp.c.prior.base.writecsv(ROOT/'cusp_confirmation_pathwise.csv',rows);cp.c.prior.base.writecsv(ROOT/'cusp_confirmation_summary.csv',summarize(rows))
    result=dict(status='completed',latent_paths=len(jobs),observations=len(jobs)*len(P['noises']),records=len(rows),seconds=time.time()-start)
    (ROOT/'cusp_confirmation_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
