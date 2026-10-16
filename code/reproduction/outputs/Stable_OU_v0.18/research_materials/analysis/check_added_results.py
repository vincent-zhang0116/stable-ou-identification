"""Numerical verification of sensitivity formulas and simulation coverage records."""
from pathlib import Path
import csv,json,sys
import numpy as np
from scipy.optimize._numdiff import approx_derivative
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'original_v013/source'))
import experiment as ex

def main():
    rng=np.random.default_rng(202610150916);det_errors=[];scalar_violations=0
    for _ in range(200):
        k=np.exp(rng.uniform(-1,1));a=rng.uniform(1.1,1.9);C=np.exp(rng.uniform(-1,1));r=np.exp(-.25*k);u=.3;t=2**a;c=t-4
        for d in [-1,1]:
            def shape(s):return (1+s)**a-1-s**a if d==1 else 1+s**a-(1-s)**a
            def prime(s):return a*((1+s)**(a-1)-s**(a-1)) if d==1 else a*(s**(a-1)+(1-s)**(a-1))
            z=C*u**a*shape(r);q=shape(r*r)/shape(r)
            qp=q/r*(2*r*r*prime(r*r)/shape(r*r)-r*prime(r)/shape(r))
            expected=(-d)*u**4*c**3*z*z*(z/C)*t*np.log(2)*.25*r*qp
            def model(v):
                kk,aa,cc,q1,q2=v;rr=np.exp(-.25*kk);rs=np.array([rr,rr,rr,rr*rr,rr*rr]);us=np.array([u,2*u,4*u,u,2*u])
                mm=(1+rs)**aa-1-rs**aa if d==1 else 1+rs**aa-(1-rs)**aa
                return -d*cc*us**aa*mm-d*np.array([q1,q1,q1,q2,q2])*us**2
            actual=np.linalg.det(approx_derivative(model,np.array([k,a,C,.2,-.1]),method='3-point'))
            det_errors.append(abs(actual-expected)/abs(expected))
        r1,r2=np.exp(-rng.uniform(.02,2,2));alpha=a
        ff=lambda x:(1+x)**alpha-1-x**alpha
        q1,q2=ff(r1*r1)/ff(r1),ff(r2*r2)/ff(r2)
        scalar_violations+=abs(np.log(r1/r2))>abs(np.log(q1/q2))/(alpha-1)+1e-10
    assert max(det_errors)<2e-5 and not scalar_violations
    # Compare analytic log-coordinate Jacobian with finite differences over decreasing frequencies.
    frequency=[];theta=np.array([1.,1.8,.8]);eta=np.array([0.,1.8,np.log(.8)])
    for u in [.5,.25,.125,.0625,.01,.001]:
        J=approx_derivative(lambda e:ex.model([np.exp(e[0]),e[1],np.exp(e[2])],.05,u)[0],eta,method='3-point')
        sv=np.linalg.svd(J,compute_uv=False);frequency.append(dict(u=u,smin=float(sv[-1]),condition=float(sv[0]/sv[-1])))
    # Re-evaluate the new study's coverage decisions from saved numerical records.
    rows=list(csv.DictReader((ROOT/'new_experiments/central_pathwise.csv').open()))
    by={(int(r['path_id']),r['noise'],int(r['n']),r['rule']):r for r in rows if r['rule']!='moment_drift'}
    mismatches=[]
    for line in (ROOT/'new_experiments/central_features.jsonl').open():
        fr=json.loads(line);r=by[tuple(fr['key'])];u=float(r['u']);n=int(r['n'])
        for kind,critical in [('hac',ex.chi2.ppf(.95,3)),('batch',3*19/17*ex.f.ppf(.95,3,17))]:
            cover=False;G=fr[kind]
            if G is not None:
                delta=np.array(fr['interaction'])-ex.model(theta,.05,u)[0]
                cover=n*delta@np.linalg.solve(np.array(G),delta)<=critical
            if bool(int(r[kind+'_covered']))!=bool(cover):mismatches.append([fr['key'],kind])
    assert not mismatches
    result=dict(five_determinants_checked=len(det_errors),max_relative_determinant_error=max(det_errors),scalar_bound_checks=200,scalar_bound_violations=scalar_violations,
                new_feature_records=len(by),new_coverage_mismatches=mismatches,frequency_jacobian_diagnostics=frequency,
                scope='Finite-difference and record-level numerical checks; analytical proofs are provided in the article.')
    result['scalar_bound_violations']=int(result['scalar_bound_violations'])
    (ROOT/'verification/added_results_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
