"""Scientific validation on deterministic population features and paired development records."""
from pathlib import Path
import json,csv,sys,hashlib
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import benchmark as b
R=Path(__file__).resolve().parents[1];cfg=json.loads(Path(__file__).with_name('design.json').read_text(encoding='utf8'));rows=[];grad=[]
for kind in ['three','nine','thirtysix','profile']:
    cells=b.grid(kind,.125,cfg)
    for theta in [[1,1.8,.8],[.3,1.12,4],[4,1.95,.1]]:
        x=np.array([np.log(theta[0]),theta[1]]);H,J=b.kernels(x,cells,.05);num=approx_derivative(lambda z:b.kernels(z,cells,.05)[0],x,method='3-point');err=float(np.max(abs(num-J)));grad.append({'grid':kind,'theta':theta,'max_abs_gradient_error':err});assert err<1e-8
        if kind not in ['nine','thirtysix','profile']:continue
        d=theta[2]*H
        if kind=='profile':
            cs=np.array(cells);d+=.24*(-cs[:,1]*cs[:,2])*(cs[:,0]==5)+.08*(-cs[:,1]*cs[:,2])*(cs[:,0]==10)
        th,status,meta=b.md_estimate(d,cells,.05,cfg['box'],cfg,profile=kind=='profile')
        relative=float(np.max(abs((th[:3]-theta)/theta))) if th is not None else float('inf');rows.append({'grid':kind,'theta':theta,'status':status,'estimate':None if th is None else th.tolist(),'max_relative_parameter_error':relative,'criterion':meta.get('criterion')});assert th is not None and meta['criterion']<1e-9 and relative<1e-3
# Population iid-Gaussian and iid-Cauchy factors cancel without substituting estimators.
cancel=[]
for alpha in [1.15,1.8,1.95]:
    k,C,h=1,.8,.05
    for ell,u,v in [(5,.25,-.25),(5,.5,-.5),(10,.25,-.25),(1,.125,.0625)]:
        r=np.exp(-k*h*ell)
        for name,noise in [('gaussian',lambda t:-2*t*t),('cauchy',lambda t:-.25*abs(t))]:
            joint=-C*abs(u+r*v)**alpha-C*(1-r**alpha)*abs(v)**alpha+noise(u)+noise(v)
            marg1=-C*abs(u)**alpha+noise(u);marg2=-C*abs(v)**alpha+noise(v)
            H=b.kernels(np.array([np.log(k),alpha]),[(ell,u,v)],h)[0][0];error=abs(joint-marg1-marg2-C*H);cancel.append(error);assert error<1e-14
with (Path(__file__).parent/'development/central_pathwise.csv').open(encoding='utf8',newline='') as f:dev=list(csv.DictReader(f))
keys={tuple(r[k] for k in ['group','path_id','n','noise']):[] for r in dev}
for r in dev:keys[tuple(r[k] for k in ['group','path_id','n','noise'])].append(r)
pairing=[]
for key,rr in keys.items():
    assert len(rr)==4 and len({r['observation_sha256'] for r in rr})==1
    primary=[r for r in rr if r['method']!='three_native'];assert len({r['shared_u'] for r in primary})==1
    pairing.append(key)
source=Path(b.__file__);report={'population_recovery_checks':rows,'analytic_gradient_checks':grad,'iid_cancellation_checks':len(cancel),'max_cancellation_error':max(cancel),'development_paired_records':len(dev),'development_observation_sets':len(pairing),'paired_hash_and_frequency_checks_passed':True,'truth_not_supplied_to_estimators':True,'all_assertions_passed':True,'scope':'Deterministic validation and development-set invariants; not confirmatory performance selection.'}
(Path(__file__).parent/'development/validation.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({'population_max_relative_error':max(r['max_relative_parameter_error'] for r in rows),'max_gradient_abs_error':max(r['max_abs_gradient_error'] for r in grad),'paired_observation_sets':len(pairing),'all_assertions_passed':True},indent=2))
