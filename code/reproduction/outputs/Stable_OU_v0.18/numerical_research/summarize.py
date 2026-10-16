from pathlib import Path
from collections import defaultdict,Counter
import csv,json,itertools,hashlib,math
import numpy as np
R=Path(__file__).resolve().parent;cfg=json.loads((R/'design.json').read_text(encoding='utf8'))
def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(p,rs):
    with p.open('w',encoding='utf8',newline='') as f:
        keys=list(dict.fromkeys(k for r in rs for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rs)
def se(x):return float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return mid-half,mid+half
def summarize(rows,keys):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        N=len(rs);S=sum(int(x['success']) for x in rs);B=sum(x.get('boundary')=='True' for x in rs);lo,hi=wilson(S,N);loss=np.array([float(x['joint_loss']) for x in rs]);d=dict(zip(keys,key));d.update(N=N,returned=S,return_fraction=S/N,return_low=lo,return_high=hi,boundary=B,boundary_fraction=B/N,joint_loss=float(loss.mean()),joint_loss_MCSE=se(loss),median_total_method_seconds=float(np.median([float(x['total_method_seconds']) for x in rs if x.get('total_method_seconds') not in ['',None]])),failure_types=json.dumps(dict(Counter(x['status'] for x in rs)),sort_keys=True))
        if rs[0].get('five_loss') not in [None,'']:
            x=np.array([float(r['five_loss']) for r in rs]);d.update(five_loss=float(x.mean()),five_loss_MCSE=se(x))
        for p in ['k','a','c','q5','q10']:
            es=np.array([float(x['error_'+p]) for x in rs if x.get('error_'+p) not in ['',None]])
            d['bias_'+p]=float(es.mean()) if len(es) else None;d['rmse_'+p]=float(np.sqrt(np.mean(es*es))) if len(es) else None
        out.append(d)
    return out
def influence(rs,p,metric):
    success=np.array([int(r['success']) for r in rs],dtype=float);x=np.array([float(r['error_'+p]) if r.get('error_'+p) not in ['',None] else 0 for r in rs]);q=success.mean()
    if q==0:return None,None
    if metric=='bias':v=float((success*x).mean()/q);return v,success*(x-v)/q
    v=float(np.sqrt((success*x*x).mean()/q));return v,np.zeros(len(rs)) if v==0 else success*(x*x-v*v)/(2*v*q)
def paired(rows,keys,comparisons):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        by={(r['path_id'],r['method']):r for r in rs};pids=sorted({r['path_id'] for r in rs},key=int)
        for new,base in comparisons:
            A=[by[p,new] for p in pids];B=[by[p,base] for p in pids]
            assert all(a['observation_sha256']==b['observation_sha256'] for a,b in zip(A,B))
            metrics=['success','joint_loss']+(['five_loss'] if A[0].get('five_loss') not in ['',None] else [])
            for m in metrics:
                dif=np.array([float(a[m])-float(b[m]) for a,b in zip(A,B)]);mean=float(dif.mean());err=se(dif);out.append(dict(zip(keys,key),comparison=new+' minus '+base,metric=m,N=len(dif),difference=mean,MCSE=err,MC_low=mean-1.959963984540054*err,MC_high=mean+1.959963984540054*err))
            for p in ['k','a','c','q5','q10']:
                for metric in ['bias','rmse']:
                    av,ai=influence(A,p,metric);bv,bi=influence(B,p,metric)
                    if av is None or bv is None or all(r.get('error_'+p) in ['',None] for r in A+B):continue
                    err=se(ai-bi);out.append(dict(zip(keys,key),comparison=new+' minus '+base,metric='conditional_'+metric+'_'+p,N=len(A),difference=av-bv,MCSE=err,MC_low=av-bv-1.959963984540054*err,MC_high=av-bv+1.959963984540054*err))
            common=[(a,b) for a,b in zip(A,B) if a['success']=='1' and b['success']=='1']
            for p in ['k','a','c','q5','q10']:
                valid=[(a,b) for a,b in common if a.get('error_'+p) not in ['',None] and b.get('error_'+p) not in ['',None]]
                if valid:
                    lossdiff=np.array([float(a['error_'+p])**2-float(b['error_'+p])**2 for a,b in valid]);out.append(dict(zip(keys,key),comparison=new+' minus '+base,metric='common_success_mse_'+p,N=len(valid),difference=float(lossdiff.mean()),MCSE=se(lossdiff)))
    return out
def main():
    dest=R/'confirmation';central=read(dest/'central_pathwise.csv');gaussian=read(dest/'gaussian_pathwise.csv')
    assert len(central)==48000 and len(gaussian)==4000
    assert len({(r['group'],r['path_id']) for r in central})==1500 and len({(r['alpha'],r['path_id']) for r in gaussian})==1000
    ckeys=['group','noise','n'];gkeys=['alpha','noise_scale'];cs=summarize(central,ckeys+['method']);gs=summarize(gaussian,gkeys+['method'])
    cp=paired(central,ckeys,[('nine_md','three_shared'),('thirtysix_md','three_shared'),('thirtysix_md','nine_md'),('three_native','three_shared')]);gp=paired(gaussian,gkeys,[('twelve_profile','five_exact')])
    for n,rs in [('central_summary.csv',cs),('gaussian_summary.csv',gs),('central_paired.csv',cp),('gaussian_paired.csv',gp)]:write(dest/n,rs)
    status=Counter(r['status'] for r in central+gaussian);audit=[r for r in central if r.get('audit_criterion_improvement') not in ['',None]];large=[r for r in audit if float(r['audit_criterion_improvement'])>1e-6]
    availability=[]
    for alpha,scale in sorted({(r['alpha'],r['noise_scale']) for r in gaussian}):
        rr=[r for r in gaussian if r['alpha']==alpha and r['noise_scale']==scale];rr=[r for r in rr if float(r['cf_min'])>cfg['cf_floor']]
        # Intersection of method availability, not selected successful fits.
        by=defaultdict(list)
        for r in rr:by[r['path_id']].append(r)
        common=[r for rs in by.values() if len(rs)==2 for r in rs];availability.extend(summarize(common,gkeys+['method']))
    write(dest/'gaussian_shared_availability_summary.csv',availability)
    report={'confirmation_latent_paths':2500,'iid_rows':len(central),'gaussian_rows':len(gaussian),'summary_rows':{'iid':len(cs),'gaussian':len(gs)},'paired_comparison_rows':{'iid':len(cp),'gaussian':len(gp)},'status_counts':dict(status),'optimizer_audit_rows':len(audit),'audit_criterion_improvements_above_1e_minus_6':len(large),'max_audit_criterion_improvement':max(float(r['audit_criterion_improvement']) for r in audit),'max_audit_parameter_distance':max(float(r['audit_parameter_distance']) for r in audit),'confirmation_estimates_replaced_by_audit':0,'source_and_design_unchanged':True,'all_attempts_retained':True}
    (dest/'analysis_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
    print('CENTRAL N8000')
    for r in cs:
        if r['group']=='central' and r['n']=='8000':print(json.dumps({k:r[k] for k in ['noise','method','returned','boundary','joint_loss','rmse_k','rmse_a','rmse_c']}))
    print('GAUSSIAN')
    for r in gs:print(json.dumps({k:r[k] for k in ['alpha','noise_scale','method','returned','boundary','joint_loss','five_loss','rmse_c','rmse_q5']}))
if __name__=='__main__':main()
