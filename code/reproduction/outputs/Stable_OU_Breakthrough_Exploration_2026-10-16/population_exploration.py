from pathlib import Path
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np
import research_core as c
ROOT=Path(__file__).parent
if __name__=='__main__':
    assert not (ROOT/'population.csv').exists()
    rows=[]
    for a in [.7,1.,1.8,1.95,2.]:
      for k in [1.,4.]:
       for noise in ['gaussian','exponential','student3','stable12']:
        for scale in [1.,2.]:
         for j in range(13):
          u=.5*2.**(-j)
          for depth in [0,1,2]:
            B=c.population_B(k,a,5,u,noise,scale,depth);estimate,status=c.inverse(np.array([float(x) for x in B]),depth,5,u)
            row=dict(alpha=a,kappa=k,noise=noise,noise_scale=scale,u=u,depth=depth,status=status,B=[str(x) for x in B])
            if estimate is not None:row.update(kappa_hat=estimate[0],alpha_hat=estimate[1],C_hat=estimate[2],drift_relative_error=abs(estimate[0]/k-1),alpha_absolute_error=abs(estimate[1]-a),scale_relative_error=abs(estimate[2]/.8-1))
            rows.append(row)
    c.prior.base.writecsv(ROOT/'population.csv',rows)
    (ROOT/'population_manifest.json').write_text(json.dumps(dict(rows=len(rows),precision=70,scope='Population formulas only, inspected before simulation protocol; no empirical paths or statistical coverage.'),indent=2),encoding='utf8')
    for noise in ['gaussian','exponential','student3','stable12']:
        for depth in [0,1,2]:
            v=[r for r in rows if r['alpha']==1.95 and r['kappa']==1 and r['noise']==noise and r['noise_scale']==2 and r['depth']==depth]
            print(noise,depth,[(r['u'],r['status'],r.get('kappa_hat'),r.get('alpha_hat')) for r in v[::3]])
