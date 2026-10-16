from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal as D,localcontext
import ast,csv,json,hashlib,time,sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
import smooth_noise_core as c
import simulation as sim
import cusp_pilot as cp
import cusp_confirmation as conf
ROOT=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
yes=lambda x:x is True or x=='True'
def near(a,b):np.testing.assert_allclose(float(a),float(b),rtol=1e-9,atol=1e-11)
def replay_observation(job):
    a,k,n,seed,xh,noise_records=job;x=c.prior.latent(seed,k,a,n)
    assert c.prior.base.old.sha_array(x)==xh
    iid=.1*np.random.default_rng([*seed[:-1],201]).standard_cauchy(n+1)
    for noise,ni,yh in noise_records:
        y=x+iid+c.noise_path([*seed[:-1],210+ni],n,noise)
        assert c.prior.base.old.sha_array(y)==yh
    return len(noise_records)
def key(p):return tuple(p[k] for k in ['alpha','kappa','n','path_id','noise','method'])
def main():
    start=time.time();preserved={}
    for name in ['Stable_OU_v0.18','Stable_OU_Research_Exploration_2026-10-15','Stable_OU_Further_Derivation_2026-10-15','Stable_OU_Additional_Paths_2026-10-16','Stable_OU_Research_Validation_2026-10-16','Stable_OU_Review_and_Intervals_2026-10-16','Stable_OU_Priority_and_Inference_2026-10-16']:
        folder=ROOT.parent/name;checks=json.loads((folder/'SHA256SUMS.json').read_text(encoding='utf8'))
        for rel,digest in checks.items():assert sha(folder/rel)==digest,(name,rel)
        preserved[name]=len(checks)
    for path in ROOT.glob('*.py'):ast.parse(path.read_text(encoding='utf8'))
    for path in ROOT.glob('*freeze.json'):
        for rel,digest in json.loads(path.read_text())['files'].items():assert sha(ROOT/rel)==digest
    hashes=set();latent=set();records=0;observations=0;detailed=0;summary_count=0
    original_cp=json.loads((ROOT/'cusp_protocol.json').read_text())
    for stage,module,protocol in [('simulation',sim,sim.P),('cusp',cp,original_cp),('cusp_confirmation',conf,conf.P)]:
        rows=list(csv.DictReader((ROOT/(stage+'_pathwise.csv')).open(encoding='utf8')))
        for r in rows:
            for k in ['alpha','kappa']:r[k]=float(r[k])
            for k in ['n','path_id']:r[k]=int(r[k])
            for k in ['loss','drift_relative_error','alpha_absolute_error','scale_relative_error','kappa_hat','alpha_hat','C_hat','cf_min']:
                if r.get(k):r[k]=float(r[k])
            r['accurate']=yes(r['accurate'])
            if 'boundary' in r:r['boundary']=yes(r['boundary'])
            if stage=='simulation':
                B=np.array(ast.literal_eval(r['B']));th,status=c.inverse(B,int(r['depth']),5,float(r['u']))
                assert status==r['status']
                if th is None:near(r['loss'],1);assert not r['accurate']
                else:
                    for k,v in zip(['kappa_hat','alpha_hat','C_hat'],th):near(r[k],v)
                    kh,ah,Ch=th;k=r['kappa'];a=r['alpha']
                    loss=min(1.,((np.log(kh/k)/np.log(160))**2+((ah-a)/1.8)**2+(np.log(Ch/.8)/np.log(1000))**2)/3)
                    near(loss,r['loss']);assert r['accurate']==bool(abs(kh/k-1)<=.25 and abs(ah-a)<=.1 and abs(Ch/.8-1)<=.5)
            elif r['status']=='returned':
                err=abs(r['kappa_hat']/r['kappa']-1);near(err,r['drift_relative_error']);assert r['accurate']==bool(err<=.25)
            else:assert not r['accurate']
            records+=1
        expected=module.summarize(rows);saved=list(csv.DictReader((ROOT/(stage+'_summary.csv')).open(encoding='utf8')))
        assert len(expected)==len(saved)
        for a,b in zip(expected,saved):
            for k,v in a.items():
                if v is None:assert b[k] in ['', 'None']
                elif isinstance(v,(float,int)):near(v,b[k])
                else:assert str(v)==b[k]
            summary_count+=1
        lookup={key(r):r for r in rows};obs={}
        for r in rows:obs.setdefault((r['alpha'],r['kappa'],r['n'],r['path_id']),{})[r['noise']]=r
        jobs=[]
        for (a,k,n,pid),vv in obs.items():
            first=next(iter(vv.values()));seed=ast.literal_eval(first['latent_seed']);xh=first['latent_sha256'];latent.add(xh)
            noise_records=[(noise,protocol['noises'].index(noise),r['observation_sha256']) for noise,r in vv.items()]
            hashes.update(yh for noise,ni,yh in noise_records);jobs.append((a,k,n,seed,xh,noise_records))
            starts=protocol.get('path_id_starts',[protocol.get('path_id_start')])
            if pid in starts:
                ai=protocol['alphas'].index(a)
                if stage=='simulation':rr=module.worker((ai,protocol['kappas'].index(k),protocol['n'].index(n),pid))
                elif stage=='cusp':cp.P=original_cp;rr=cp.worker((ai,pid))
                else:rr=conf.worker((ai,protocol['kappas'].index(k),pid))
                for r in rr:
                    target=lookup[key(r)];assert r['status']==target['status'] and r['accurate']==target['accurate']
                    if stage=='simulation':np.testing.assert_allclose(r['B'],ast.literal_eval(target['B']),rtol=1e-11,atol=1e-12)
                    if r['status']=='returned':near(r['kappa_hat'],target['kappa_hat'])
                    assert r['observation_sha256']==target['observation_sha256'];detailed+=1
        with ProcessPoolExecutor(max_workers=4) as pool:observations+=sum(pool.map(replay_observation,jobs,chunksize=5))
        print(json.dumps(dict(stage=stage,records=records,observations_replayed=observations)),flush=True)
    prior=set()
    for name in ['Stable_OU_Research_Validation_2026-10-16','Stable_OU_Review_and_Intervals_2026-10-16','Stable_OU_Priority_and_Inference_2026-10-16']:
        for path in (ROOT.parent/name).rglob('*pathwise.csv'):
            for r in csv.DictReader(path.open(encoding='utf8')):
                prior.update(r[k] for k in ['observation_sha256','latent_sha256'] if r.get(k))
    assert not (hashes|latent)&prior
    pop=list(csv.DictReader((ROOT/'population.csv').open(encoding='utf8')));population_replays=0
    for r in pop[::17]:
        B=c.population_B(float(r['kappa']),float(r['alpha']),5,float(r['u']),r['noise'],float(r['noise_scale']),int(r['depth']))
        assert [str(x) for x in B]==ast.literal_eval(r['B']);population_replays+=1
    # Check actual nuisance remainders, independently of the inverse estimator.
    slopes=[]
    with localcontext() as ctx:
      ctx.prec=70
      for noise,depth,expected_order in [('exponential',1,4),('exponential',2,6),('student3',1,3),('student3',2,3)]:
        rem=[]
        for u in [.00390625,.001953125]:
            full=c.population_B(1,1.8,5,u,noise,1,depth);clean=c.population_B(1,1.8,5,u,'gaussian',0,depth)
            rem.append(max(abs(x-y) for x,y in zip(full,clean)))
        order=float((rem[0]/rem[1]).ln()/D(2).ln())
        assert abs(order-expected_order)<.06,(noise,depth,order)
        slopes.append(dict(noise=noise,depth=depth,nuisance_order=order))
    # Independent integral check of the fourth-difference profile.
    fourier=0
    for a in [1.,1.8]:
      K=2*gamma(a+1)*np.sin(np.pi*a/2)/np.pi
      for x in [0.,.3,2.]:
        integral,error=quad(lambda t:16*K*np.sin(t/2)**4*np.cos(t*x)/t**(1+a),0,1000,epsabs=1e-8,limit=4000)
        exact=-(abs(x+2)**a-4*abs(x+1)**a+6*abs(x)**a-4*abs(x-1)**a+abs(x-2)**a)
        assert abs(integral-exact)<=16*K/(a*1000**a)+5e-7;fourier+=1
    assert len(latent)==1350 and len(hashes)==5010 and observations==5010 and records==27720
    report=dict(status='passed',preserved_checksums=preserved,new_latent_paths=len(latent),new_observations=len(hashes),observation_hashes_replayed=observations,records_checked=records,summary_cells_recomputed=summary_count,detailed_feature_or_cusp_replays=detailed,prior_overlap=0,population_records=len(pop),population_70_digit_replays=population_replays,nuisance_order_checks=slopes,fourier_profile_integral_checks=fourier,endpoint_checks=json.loads((ROOT/'endpoint_cusp_manifest.json').read_text()),seconds=time.time()-start,scope='Implementation and deterministic example checks; mathematical proofs internally audited, no external review or statistical coverage certification.')
    (ROOT/'Verification.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
