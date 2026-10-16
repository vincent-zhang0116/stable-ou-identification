from pathlib import Path
import csv,json,hashlib,math,collections
import numpy as np
R=Path(__file__).resolve().parents[1]
O=R/'outputs/Stable_OU_v0.20';D=R/'outputs/Stable_OU_v0.18/numerical_research'
def read(p):
    with p.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (O/'supplement'/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(x):
    return float(np.mean(x)),float(np.std(x,ddof=1)/np.sqrt(len(x)))
def audit():
    cfg=json.loads((D/'design.json').read_text());central=read(D/'confirmation/central_pathwise.csv');gauss=read(D/'confirmation/gaussian_pathwise.csv')
    truth={g['name']:g['theta'] for g in cfg['groups']};spans=np.array([np.log(100),.94,np.log(1000)])
    choices=[('squared_spans_f1',2,spans,1.),('absolute_spans_f1',1,spans,1.),('squared_spans_f025',2,spans,.25),('squared_spans_f05',2,spans,.5),('squared_spans_f2',2,spans,2.),('squared_index_weighted',2,np.array([1,.2,1]),1.),('squared_unit_scales',2,np.ones(3),1.)]
    sensitivity=[]
    for name,p,scale,failure in choices:
        for noise in cfg['noises']:
            rr=[r for r in central if r['group']=='central' and r['n']=='8000' and r['noise']==noise]
            by={}
            for r in rr:
                t=truth[r['group']]
                if r['success']=='1':
                    e=np.array([math.log(float(r['estimate_k'])/t[0]),float(r['estimate_a'])-t[1],math.log(float(r['estimate_c'])/t[2])]);loss=min(1.,float(np.sum(abs(e/scale)**p)))
                else:loss=failure
                by[r['path_id'],r['method']]=loss
            ids=sorted({r['path_id'] for r in rr})
            for method in ['nine_md','thirtysix_md']:
                v=[by[i,method]-by[i,'three_native'] for i in ids];mean,se=stats(v)
                sensitivity.append(dict(loss=name,noise=noise,method=method,baseline='three_native',N=len(ids),difference=mean,MCSE=se,MC_low=mean-1.96*se,MC_high=mean+1.96*se))
    write('Loss_Sensitivity.csv',sensitivity)
    # Match the two historic table checks independently of copied PDF figures.
    expected=[[-.0080,-.0111,-.0518,-.5570],[-.0004,-.0021,-.0727,-.3209],[-.0005,-.0006,-.0173,-.0290],[-.0030,-.0041,-.0288,-.2050],[-.0180,-.0251,-.0978,-1.2610],[.0324,.0380,-.1262,-.0429],[-.0081,-.0131,-.0846,-.1755]]
    for j,(name,_,_,_) in enumerate(choices):
        vals=[r['difference'] for r in sensitivity if r['loss']==name and r['method']=='thirtysix_md']
        np.testing.assert_allclose(vals,expected[j],rtol=0,atol=.000051)
    faces=[];combos=[];conditional=[];optimizer=[]
    for stage,rows,keys in [('iid',central,['group','noise','n','method']),('gaussian',gauss,['alpha','noise_scale','method'])]:
        groups=collections.defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        for key,rr in groups.items():
            counts=collections.Counter();sets=collections.Counter();returned=[r for r in rr if r['success']=='1']
            for r in returned:
                x=[math.log(float(r['estimate_k'])),float(r['estimate_a']),float(r['estimate_c'])]
                lows=[math.log(cfg['box'][0][0]),cfg['box'][1][0],cfg['box'][2][0]];highs=[math.log(cfg['box'][0][1]),cfg['box'][1][1],cfg['box'][2][1]]
                active=[]
                for i,p in enumerate(['kappa','alpha','C']):
                    for side,bound in [('lower',lows[i]),('upper',highs[i])]:
                        if abs(x[i]-bound)<1e-6:counts[p+'_'+side]+=1;active.append(p+'_'+side)
                # Three/five direct methods have no constrained optimizer flag;
                # calculate their faces but separately preserve recorded status.
                if active:counts['any']+=1;sets[' & '.join(active)]+=1
                if len(active)>1:counts['multiple']+=1
            header=dict(stage=stage,**dict(zip(keys,key)))
            faces.append(dict(header,N=len(rr),returned=len(returned),recorded_boundary=sum(r['boundary']=='True' for r in rr),**{x:counts[x] for x in ['any','kappa_lower','kappa_upper','alpha_lower','alpha_upper','C_lower','C_upper','multiple']}))
            for k,n in sets.items():combos.append(dict(header,combination=k,count=n))
            for p in ['k','a','c']+(['q5','q10'] if stage=='gaussian' else []):
                for subset,sub in [('all_returned',returned),('interior_returned',[r for r in returned if r['boundary']!='True']),('boundary_returned',[r for r in returned if r['boundary']=='True'])]:
                    e=[float(r['error_'+p]) for r in sub if r['error_'+p]!='']
                    conditional.append(dict(header,parameter=p,subset=subset,N_attempts=len(rr),denominator=len(e),bias=np.mean(e) if e else '',RMSE=np.sqrt(np.mean(np.square(e))) if e else ''))
        fits=[r for r in rows if r.get('optimizer_runs')]
        optimizer.append(dict(stage=stage,fits=len(fits),local_runs=sum(int(r['optimizer_runs']) for r in fits),converged=sum(int(r['optimizer_converged']) for r in fits),fits_with_nonconverged_start=sum(int(r['optimizer_converged'])<int(r['optimizer_runs']) for r in fits),fits_spread_over_1e8=sum(float(r['local_criterion_spread'])>1e-8 for r in fits),max_spread=max(float(r['local_criterion_spread']) for r in fits)))
    # Heterogeneous group columns use a common schema.
    def homogeneous(rows):
        columns=list(dict.fromkeys(k for r in rows for k in r));return [{k:r.get(k,'') for k in columns} for r in rows]
    write('Boundary_Faces.csv',homogeneous(faces));write('Boundary_Combinations.csv',homogeneous(combos));write('Conditional_Parameter_Errors.csv',homogeneous(conditional));write('Optimizer_Diagnostics.csv',optimizer)
    for stage,noise,method,expected_counts in [('iid','uniform_zero','nine_md',[274,104,19,11,179,1,0,39]),('iid','uniform_zero','thirtysix_md',[212,113,20,5,113,0,0,39]),('gaussian','1.0','twelve_profile',[178,39,64,24,127,67,0,100]),('gaussian','2.0','twelve_profile',[194,66,88,9,153,66,6,136])]:
        matching=[f for f in faces if f['stage']==stage and f['method']==method and ((stage=='iid' and f['noise']==noise and f['group']=='central' and f['n']=='8000') or (stage=='gaussian' and f['alpha']=='1.98' and f['noise_scale']==noise))]
        assert len(matching)==1,(stage,noise,method)
        f=matching[0];assert [f[k] for k in ['any','kappa_lower','kappa_upper','alpha_lower','alpha_upper','C_lower','C_upper','multiple']]==expected_counts
    pairs=[]
    for group in ['central','near_one','fast']:
        for noise in cfg['noises']:
            rr=[r for r in central if r['group']==group and r['n']=='8000' and r['noise']==noise];by={(r['path_id'],r['method']):r for r in rr};ids=sorted({r['path_id'] for r in rr})
            for baseline in ['three_shared','three_native']:
                for method in ['nine_md','thirtysix_md']:
                    x=[float(by[i,method]['joint_loss'])-float(by[i,baseline]['joint_loss']) for i in ids];mean,se=stats(x)
                    pairs.append(dict(group=group,noise=noise,baseline=baseline,method=method,N=len(ids),difference=mean,MCSE=se,MC_low=mean-1.96*se,MC_high=mean+1.96*se))
    write('Dual_Baseline_Figure_Data.csv',pairs)
    def put(x,y,s,anchor='c'):return rf'\put({x:.3f},{y:.3f}){{\makebox(0,0)[{anchor}]{{\scriptsize {s}}}}}'
    def line(x,y,xx,yy):return rf'\qbezier({x:.3f},{y:.3f})({(x+xx)/2:.3f},{(y+yy)/2:.3f})({xx:.3f},{yy:.3f})'
    fig=[r'\begin{figure}[htbp]\centering',r'\setlength{\unitlength}{1pt}',r'\begin{picture}(450,330)',r'\linethickness{.55pt}']
    for ri,baseline in enumerate(['three_shared','three_native']):
        for ci,group in enumerate(['central','near_one','fast']):
            x0=34+ci*148;y0=187-ri*150;w=112;hh=105
            ym=lambda v:y0+(v+1.05)/1.2*hh
            fig += [line(x0,y0,x0+w,y0),line(x0,y0,x0,y0+hh),put(x0+w/2,y0+hh+13,['Central',r'$\alpha=1.15$',r'$\kappa=4$'][ci])]
            for tick in [-1,-.5,0]:fig += [line(x0-2,ym(tick),x0,ym(tick)),put(x0-5,ym(tick),str(tick),'r')]
            for j in range(8):fig.append(line(x0+j*w/8,ym(0),x0+(j+.5)*w/8,ym(0)))
            for ni,noise in enumerate(cfg['noises']):
                x=x0+12+ni*29
                fig.append(put(x,y0-12,['Cl','Ca','G','U'][ni]))
                for mi,method in enumerate(['nine_md','thirtysix_md']):
                    z=next(r for r in pairs if all(r[k]==v for k,v in [('group',group),('noise',noise),('baseline',baseline),('method',method)]));xx=x+(-2.7 if mi==0 else 2.7)
                    for v in [z['MC_low'],z['MC_high']]:assert -1.05<=v<=.15
                    fig += [line(xx,ym(z['MC_low']),xx,ym(z['MC_high'])),line(xx-1.8,ym(z['MC_low']),xx+1.8,ym(z['MC_low'])),line(xx-1.8,ym(z['MC_high']),xx+1.8,ym(z['MC_high'])),put(xx,ym(z['difference']),r'$\circ$' if mi==0 else r'$\square$')]
            if ci==0:fig.append(put(7,y0+hh/2,r'\rotatebox{90}{MD minus '+('3S' if ri==0 else '3N')+'}'))
    fig += [put(225,8,r'$\circ$: 9 MD\qquad $\square$: 36 MD\qquad Cl: clean; Ca: Cauchy; G: Gaussian; U: uniform'),r'\end{picture}',r'\caption{Paired all-attempt loss differences at $n=8{,}000$, relative to shared-frequency 3S (top) and native-frequency 3N (bottom). Each panel uses 500 independent latent paths per error law, with methods paired on identical observations. All panels use the same vertical scale. Bars show the mean plus or minus 1.96 paired Monte Carlo standard errors; these are pointwise descriptive intervals, not simultaneous intervals or process-parameter confidence intervals. Negative values favor MD.}\label{fig:benchmark}',r'\end{figure}']
    (O/'supplement/Figure5_Dual_Baseline.tex').write_text('\n'.join(fig),encoding='utf8')
    log=dict(stage=2,passed=True,source_csv_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'confirmation/central_pathwise.csv',D/'confirmation/gaussian_pathwise.csv']},sensitivity_contrasts=len(sensitivity),boundary_groups=len(faces),boundary_combinations=len(combos),parameter_error_rows=len(conditional),dual_baseline_contrasts=len(pairs),historical_rounded_sensitivity_values_checked=28,historical_boundary_counts_checked=32,new_fits=0,posthoc_loss_evaluation=True,optimizer=optimizer)
    (O/'verification/Numerical_Restoration.json').write_text(json.dumps(log,indent=2),encoding='utf8');print(json.dumps(log,indent=2))
if __name__=='__main__':audit()
